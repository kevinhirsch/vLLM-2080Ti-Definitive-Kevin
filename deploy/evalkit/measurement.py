"""Shared client measurements. Arrival gaps are SSE events, never token ITL.

No secrets, generated content or prompt bodies are included in provenance.
The returned response stays available to the quality scorer separately.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import requests

SAMPLING_KEYS = {
    'temperature', 'top_p', 'top_k', 'min_p', 'seed', 'presence_penalty',
    'frequency_penalty', 'repetition_penalty', 'chat_template_kwargs',
}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_profile(path):
    profile = json.loads(Path(path).read_text()) if path else {}
    if not isinstance(profile, dict) or set(profile) - SAMPLING_KEYS:
        raise ValueError('Profile must be an object containing only sampling/thinking fields')
    if 'chat_template_kwargs' in profile and not isinstance(profile['chat_template_kwargs'], dict):
        raise ValueError('chat_template_kwargs must be an object')
    return profile


def merge_request(defaults, item, profile=None):
    """Explicit experiment profile wins, including on every agentic turn."""
    payload = dict(defaults)
    for layer in (item, profile or {}):
        payload.update({k: v for k, v in layer.items() if k != 'chat_template_kwargs'})
        payload['chat_template_kwargs'] = {
            **payload.get('chat_template_kwargs', {}), **layer.get('chat_template_kwargs', {})}
    return payload


def generation_error(response):
    """Quality cannot pass an error, malformed or zero-token generation."""
    if not isinstance(response, dict) or response.get('error'):
        return 'error response'
    usage = response.get('usage')
    if not isinstance(usage, dict):
        return 'missing usage object'
    tokens = usage.get('completion_tokens')
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
        return 'missing or zero completion_tokens'
    choices = response.get('choices')
    if not isinstance(choices, list) or len(choices) != 1:
        return 'expected one generated choice'
    choice = choices[0]
    if not isinstance(choice, dict):
        return 'invalid choice object'
    if choice.get('finish_reason') not in ('stop', 'length', 'tool_calls', 'function_call'):
        return 'missing or unsuccessful finish_reason'
    message = choice.get('message') or {}
    if not isinstance(message, dict):
        return 'invalid message object'
    calls = message.get('tool_calls') or []
    if calls:
        if not isinstance(calls, list):
            return 'invalid tool_calls object'
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get('id'), str) or not call['id']:
                return 'incomplete tool call ID'
            fn = call.get('function')
            if not isinstance(fn, dict) or not isinstance(fn.get('name'), str) or not fn['name']:
                return 'incomplete tool function name'
            try:
                args = json.loads(fn.get('arguments', ''))
            except (ValueError, TypeError):
                return 'incomplete or invalid tool arguments'
            if not isinstance(args, dict):
                return 'tool arguments must be a JSON object'
    if not calls and not any(isinstance(message.get(k), str) and message[k].strip() for k in ('content', 'reasoning_content')):
        return 'empty assistant generation'
    return None


def response_receipt(response):
    """Content-free generation identity; omit response/call IDs and bodies."""
    choices = response.get('choices') if isinstance(response, dict) else None
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get('message') or {'content': choice.get('text')}
    if not isinstance(message, dict):
        message = {}
    content, reasoning = message.get('content'), message.get('reasoning_content')
    tools = []
    calls = message.get('tool_calls')
    for call in calls if isinstance(calls, list) else []:
        if not isinstance(call, dict):
            continue
        fn = call.get('function') or {}
        if not isinstance(fn, dict):
            continue
        arguments = fn.get('arguments')
        try:
            arguments = json.loads(arguments)
        except (ValueError, TypeError):
            pass
        tools.append({'name': fn.get('name'), 'arguments': arguments})
    return {'finish_reason': choice.get('finish_reason'),
            'content_sha256': canonical_hash(content), 'reasoning_sha256': canonical_hash(reasoning),
            'assistant_output_sha256': canonical_hash({'content': content, 'reasoning': reasoning, 'tools': tools}),
            'has_visible_prose': isinstance(content, str) and bool(content.strip()),
            'has_reasoning': isinstance(reasoning, str) and bool(reasoning.strip()), 'tool_calls_count': len(tools)}


def quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


class StreamTimeline:
    """Assemble split tool deltas and retain a content-free arrival timeline.

    Tool-complete means name + ID + parseable argument object, not correctness
    or executed effect. A quality scorer must establish those independently.
    """
    def __init__(self, start):
        self.start = start
        self.first = dict.fromkeys(('body_byte', 'generated_delta', 'prose', 'tool_delta', 'tool_complete'))
        self.arrivals = {'prose': [], 'tool': [], 'generated': []}
        self.generated_counts = []
        self.content = []
        self.reasoning = []
        self.tools = {}
        self.tool_complete_times = {}
        self.usage = {}
        self.metrics = {}
        self.finish = None
        self.response_meta = {}
        self.done = False

    def byte(self, now):
        if self.first['body_byte'] is None:
            self.first['body_byte'] = now - self.start

    def event(self, event, now):
        if event.get('error'):
            raise ValueError('stream error event')
        self.response_meta.update({k: event[k] for k in ('id', 'model', 'created') if k in event})
        if event.get('usage'):
            self.usage = event['usage']
        if event.get('metrics'):
            self.metrics.update(event['metrics'])
        for choice in event.get('choices') or []:
            if choice.get('index', 0) != 0:
                raise ValueError('Multiple choices are unsupported by this measurement')
            delta = choice.get('delta') or {}
            content = delta.get('content') or ''
            reasoning = delta.get('reasoning_content') or ''
            calls = delta.get('tool_calls') or []
            t = now - self.start
            if content or reasoning or calls:
                if self.first['generated_delta'] is None:
                    self.first['generated_delta'] = t
                self.arrivals['generated'].append(t)
                count = (event.get('usage') or {}).get('completion_tokens')
                self.generated_counts.append(count if isinstance(count, int) and not isinstance(count, bool) else None)
            if content:
                self.content.append(content)
                if content.strip():
                    if self.first['prose'] is None:
                        self.first['prose'] = t
                    self.arrivals['prose'].append(t)
            if reasoning:
                self.reasoning.append(reasoning)
            if calls:
                if self.first['tool_delta'] is None:
                    self.first['tool_delta'] = t
                self.arrivals['tool'].append(t)
            for call in calls:
                index = call.get('index', 0)
                tool = self.tools.setdefault(index, {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                changed = any((call.get('function') or {}).get(k) for k in ('name', 'arguments')) or bool(call.get('id'))
                if changed:
                    self.tool_complete_times.pop(index, None)
                tool['id'] += call.get('id') or ''
                for key in ('name', 'arguments'):
                    tool['function'][key] += (call.get('function') or {}).get(key) or ''
                try:
                    args = json.loads(tool['function']['arguments'])
                except (ValueError, TypeError):
                    continue
                if tool['id'] and tool['function']['name'] and isinstance(args, dict):
                    self.tool_complete_times.setdefault(index, t)
            self.first['tool_complete'] = min(self.tool_complete_times.values(), default=None)
            if choice.get('finish_reason'):
                self.finish = choice['finish_reason']

    def response(self):
        message = {'role': 'assistant', 'content': ''.join(self.content) or None}
        if self.reasoning:
            message['reasoning_content'] = ''.join(self.reasoning)
        if self.tools:
            message['tool_calls'] = [self.tools[i] for i in sorted(self.tools)]
        response = {**self.response_meta, 'choices': [{'index': 0, 'message': message, 'finish_reason': self.finish}], 'usage': self.usage}
        if self.metrics:
            response['metrics'] = self.metrics
        return response

    def report(self, end):
        report = {f'first_{key}_s': value for key, value in self.first.items()}
        useful = [self.first[k] for k in ('prose', 'tool_complete') if self.first[k] is not None]
        report['first_useful_candidate_s'] = min(useful) if useful else None
        report['all_tools_complete_s'] = max(self.tool_complete_times.values()) if self.tools and len(self.tools) == len(self.tool_complete_times) else None
        report['wall_s'] = end - self.start
        report['stream_done'] = self.done
        report['generated_event_completion_tokens'] = self.generated_counts
        times = self.arrivals['generated']
        counts = self.generated_counts
        counted = len(times) > 1 and all(n is not None for n in counts) and times[-1] > times[0] and all(b >= a for a, b in zip(counts, counts[1:]))
        report['counted_generated_tokens_s'] = (counts[-1]-counts[0])/(times[-1]-times[0]) if counted else None
        report['counted_rate_semantics'] = 'server cumulative generated-token change between first/last nonempty SSE arrival; includes reasoning/tool tokens; not token ITL'
        report['gap_semantics'] = 'client SSE arrival events; bundles may contain several tokens; excludes initial wait'
        report['first_useful_semantics'] = 'non-whitespace prose or syntactically complete tool call; correctness/effect separately scored'
        for kind, arrivals in self.arrivals.items():
            gaps = [b-a for a, b in zip(arrivals, arrivals[1:])]
            report[f'{kind}_events'] = len(arrivals)
            report[f'{kind}_event_times_s'] = arrivals
            report[f'{kind}_gap_p95_s'] = quantile(gaps, .95)
            report[f'{kind}_gap_max_s'] = max(gaps) if gaps else None
            report[f'{kind}_stalls_over_250ms'] = sum(g > .25 for g in gaps)
        return report


def post_chat(base_url, payload, timeout_s=180, headers=None):
    start = time.perf_counter()
    wire = {**payload, 'stream': True, 'stream_options': {**payload.get('stream_options', {}), 'include_usage': True,
            'continuous_usage_stats': payload.get('stream_options', {}).get('continuous_usage_stats', True)}}
    request_hash = canonical_hash(wire)
    timeline = StreamTimeline(start)
    try:
        with requests.post(base_url.rstrip('/') + '/v1/chat/completions', json=wire,
                           timeout=timeout_s, headers=headers, stream=True) as response:
            if response.status_code != 200:
                return {'status': 'http_error', 'request_payload_sha256': request_hash, 'status_code': response.status_code, 'wall_s': time.perf_counter()-start}
            # read1 returns available bytes without iter_lines' default 512-byte
            # accumulation delay. Timestamp arrival before decoding an SSE frame.
            pending = b''
            frame = []
            while True:
                chunk = response.raw.read1(65536)
                if not chunk:
                    break
                now = time.perf_counter()
                timeline.byte(now)
                pending += chunk
                while b'\n' in pending:
                    line, pending = pending.split(b'\n', 1)
                    line = line.rstrip(b'\r')
                    if line:
                        if line.startswith(b'data:'):
                            frame.append(line[5:].lstrip())
                        continue
                    if not frame:
                        continue
                    data = b'\n'.join(frame)
                    frame = []
                    if data.strip() == b'[DONE]':
                        timeline.done = True
                        break
                    timeline.event(json.loads(data), now)
                if timeline.done:
                    break
            result = timeline.response()
            error = generation_error(result)
            if not timeline.done:
                error = 'stream ended without DONE'
            return {'status': 'invalid_generation' if error else 'ok', 'error': error, 'request_payload_sha256': request_hash,
                    'response': result, 'timeline': timeline.report(time.perf_counter()),
                    'wall_s': time.perf_counter()-start,
                    'route_receipts': {k: v for k, v in response.headers.items() if k.lower() in ('x-gateway-route', 'x-window-identity', 'x-route-provider', 'x-route-model', 'x-gateway-request-id')}}
    except requests.exceptions.RequestException as exc:
        return {'status': 'engine-crash', 'request_payload_sha256': request_hash, 'error': type(exc).__name__, 'wall_s': time.perf_counter()-start}
    except (ValueError, TypeError, KeyError) as exc:
        return {'status': 'invalid_generation', 'request_payload_sha256': request_hash, 'error': type(exc).__name__, 'wall_s': time.perf_counter()-start,
                'response': timeline.response(), 'timeline': timeline.report(time.perf_counter())}


def file_identity(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def git_identity(path):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(path), *args], text=True, stderr=subprocess.DEVNULL).strip()
    try:
        dirty = git('status', '--porcelain')
        return {'path': str(Path(path).resolve()), 'sha': git('rev-parse', 'HEAD'),
                'dirty': bool(dirty), 'tracked_diff_sha256': hashlib.sha256(git('diff', 'HEAD').encode()).hexdigest()}
    except (OSError, subprocess.CalledProcessError):
        return {'path': str(path), 'unavailable': True}


def runtime_identity(pid):
    """Whitelist engine flags; never record arbitrary argv or environment."""
    proc = Path('/proc') / str(pid)
    raw_argv = proc.joinpath('cmdline').read_bytes()
    args = raw_argv.decode().rstrip('\0').split('\0')
    flags = {}
    allowed = {'--model', '--generation-config', '--served-model-name', '--tensor-parallel-size', '--kv-cache-dtype', '--speculative-config', '--max-model-len', '--gdn-prefill-backend', '--per-request-spec-decode-metrics',
               '--long-prefill-token-threshold', '--max-num-partial-prefills', '--max-long-partial-prefills', '--dtype', '--gpu-memory-utilization', '--compilation-config', '--hf-overrides', '--max-num-seqs', '--max-num-batched-tokens', '--scheduling-policy', '--mamba-cache-mode', '--limit-mm-per-prompt', '--mm-processor-kwargs', '--chat-template', '--reasoning-parser', '--tool-call-parser', '--additional-config', '--kv-transfer-config',
               '--async-scheduling', '--no-async-scheduling', '--enable-chunked-prefill', '--enable-prefix-caching', '--enable-prompt-tokens-details', '--enable-auto-tool-choice'}
    for i, arg in enumerate(args):
        key, sep, value = arg.partition('=')
        if key in allowed:
            if sep:
                flags[key] = value
            else:
                values = []
                for candidate in args[i+1:]:
                    if candidate.startswith('--'):
                        break
                    values.append(candidate)
                flags[key] = values if key == '--served-model-name' else (values[0] if values else True)
    cwd = proc.joinpath('cwd').resolve()
    # Field 22 begins after the parenthesized executable name, whose spaces
    # cannot safely be split as an ordinary whitespace-delimited record.
    start_ticks = proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19]
    identity = {'pid': int(pid), 'start_ticks': start_ticks, 'source': git_identity(cwd), 'flags': flags,
                'argv_sha256': hashlib.sha256(raw_argv).hexdigest()}
    safe_env_keys = {'VLLM_SCHED_SHORT_FIRST_PREFIX_AWARE', 'VLLM_SCHED_PREFILL_SHARE_EXACT', 'VLLM_SCHED_INVARIANT_GUARD', 'VLLM_SCHED_PREFILL_SHARE', 'VLLM_SCHED_SHORT_FIRST', 'VLLM_SCHED_SHORT_FIRST_RUN', 'CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'VLLM_USE_V1', 'VLLM_TQ_GQA_CUDA', 'VLLM_TQ_GQA_GROUP', 'VLLM_TQ_GQA_BATCH_THRESHOLD', 'VLLM_TQ_GQA_BUILD_DIR', 'VLLM_K5_GDN_FUSED', 'VLLM_U2_INT4_HEAD', 'VLLM_U2_INT4_MTP', 'VLLM_MAMBA_ALIGN_RETAIN_MTP_CACHE_BLOCK', 'VLLM_PREFIX_CACHE_USE_RETAINED_MTP_BLOCK', 'VLLM_TURBOQUANT_CONTINUATION_WORKSPACE_RESERVE_TOKENS'}
    identity['runtime_env'] = {}
    for raw in proc.joinpath('environ').read_bytes().split(b'\0'):
        key, sep, value = raw.decode(errors='replace').partition('=')
        if sep and key in safe_env_keys:
            identity['runtime_env'][key] = value
    identity['provenance_limits'] = 'Process/source/config snapshot; weight manifest uses sizes/mtime, not full weight hashes; mapped native files are checked separately against the controller/release manifest; dispatched CUDA functions require a live profiling receipt'
    release_manifest = cwd / 'RELEASE.json'
    if release_manifest.is_file():
        release = json.loads(release_manifest.read_text())
        identity['source'].pop('unavailable', None)
        identity['source'].update({'identity_source': 'RELEASE.json', 'sha': release.get('sha'), 'release_id': release.get('id'),
                                   'release_manifest': file_identity(release_manifest)})
    checkpoint = flags.get('--model')
    if checkpoint:
        checkpoint = str((cwd / checkpoint).resolve()) if not Path(checkpoint).is_absolute() else checkpoint
        identity['checkpoint'] = {'root': str(Path(checkpoint).resolve()), 'metadata': [],
            'weight_files': [{'name': f.name, 'bytes': f.stat().st_size, 'mtime_ns': f.stat().st_mtime_ns}
                             for f in sorted(Path(checkpoint).glob('*.safetensors'))]}
        for name in ('config.json', 'generation_config.json', 'model.safetensors.index.json', 'tokenizer_config.json'):
            f = Path(checkpoint) / name
            if f.is_file():
                identity['checkpoint']['metadata'].append(file_identity(f))
    generation = flags.get('--generation-config')
    if generation and generation != 'vllm':
        f = Path(generation) if Path(generation).is_absolute() else cwd / generation
        if f.is_dir():
            f = f / 'generation_config.json'
        if f.is_file():
            identity['selected_generation_config'] = file_identity(f)
    template = flags.get('--chat-template')
    if isinstance(template, str) and Path(template).is_file():
        identity['chat_template'] = file_identity(template)
    identity['fingerprint'] = canonical_hash(identity)
    return identity


def spec_counter_delta(before, after, isolated=False):
    keys = ('drafts', 'draft_tokens', 'accepted_tokens')
    if before is None or after is None or any(k not in before or k not in after for k in keys):
        return {'available': False, 'reason': 'missing counters'}
    delta = {k: after[k] - before[k] for k in keys}
    if any(v < 0 for v in delta.values()):
        return {'available': False, 'reason': 'counter reset'}
    steps = delta['drafts']
    return {'available': True, **delta, 'scope': 'isolated wave' if isolated else 'shared engine; not per-request',
            'accepted_draft_tokens_per_verify': delta['accepted_tokens']/steps if steps else None,
            'estimated_emitted_tokens_per_verify_including_bonus': 1+delta['accepted_tokens']/steps if steps else None,
            'draft_acceptance_rate': delta['accepted_tokens']/delta['draft_tokens'] if delta['draft_tokens'] else None,
            'bonus_semantics': 'bonus estimate requires one bonus token per verify; EOS/grammar/partial steps can differ'}


def scrape_spec_counters(base_url):
    root = base_url.rstrip('/').removesuffix('/v1')
    metrics = {'drafts': 'vllm:spec_decode_num_drafts_total', 'draft_tokens': 'vllm:spec_decode_num_draft_tokens_total', 'accepted_tokens': 'vllm:spec_decode_num_accepted_tokens_total'}
    found = {}
    try:
        response = requests.get(root + '/metrics', timeout=10)
        response.raise_for_status()
        for line in response.text.splitlines():
            for key, name in metrics.items():
                if re.match(re.escape(name) + r'(?:\{|\s)', line):
                    found[key] = found.get(key, 0) + float(line.rsplit(None, 1)[-1])
    except (requests.RequestException, ValueError):
        return None
    return found or None


def verify_models_root(base_url, identity):
    response = requests.get(base_url.rstrip('/').removesuffix('/v1') + '/v1/models', timeout=10)
    response.raise_for_status()
    roots = {str(Path(row['root']).resolve()) for row in response.json().get('data', []) if row.get('root')}
    checkpoint = identity.get('checkpoint', {}).get('root')
    if not checkpoint or roots != {checkpoint}:
        raise ValueError('Engine /v1/models checkpoint does not match actual process --model')
    return sorted(roots)


def require_offline_window(gateway='http://127.0.0.1:8000', drained=False):
    response = requests.get(gateway + '/gateway/offline', timeout=5)
    response.raise_for_status()
    state = response.json()
    if state.get('offline') is not True or state.get('remaining_s', 0) <= 0:
        raise ValueError('A live gateway offline fence is required for direct-engine measurements')
    if drained and (state.get('local_active') or state.get('local_inflight_units')):
        raise ValueError('Accepted local requests have not drained')
    return {'offline': True, 'remaining_s': state['remaining_s']}


def _process_start(pid, proc_root):
    return (proc_root / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19]


def _process_tree(pid, proc_root):
    """Follow all thread child lists; no name/pattern matching of other jobs."""
    found, pending = set(), [pid]
    while pending:
        current = pending.pop()
        if current in found:
            continue
        found.add(current)
        tasks = proc_root / str(current) / 'task'
        if not tasks.is_dir():
            raise ValueError('Cannot attest engine descendant process tree')
        for task in tasks.iterdir():
            for child in (task / 'children').read_text().split():
                pending.append(int(child))
    return sorted(found)


def verify_loaded_native(identity, identity_file=None, manifest_file=None, proc_root=Path('/proc')):
    """Hash frozen native files and match mapped inode/device in engine workers.

    Accept immutable RELEASE.json or an explicitly supplied controller legacy
    snapshot {prod_root, so}. Never interpret the legacy tree as immutable.
    This proves mapped host files, not which CUDA function was dispatched.
    """
    import os
    root = Path(identity['source']['path']).resolve()
    pid = identity['pid']
    if _process_start(pid, proc_root) != str(identity['start_ticks']):
        raise ValueError('Engine PID/start changed before native attestation')
    attested = None
    if identity_file:
        expected = json.loads(Path(identity_file).read_text())
        if (expected.get('engine_pid') != pid or str(expected.get('pid_start')) != str(identity['start_ticks'])
                or Path(expected.get('root', '')).resolve() != root):
            raise ValueError('Controller engine identity does not match actual PID/start/root')
        actual_argv = (proc_root / str(pid) / 'cmdline').read_bytes()
        if expected.get('argv_sha256') and hashlib.sha256(actual_argv).hexdigest() != expected['argv_sha256']:
            raise ValueError('Controller command-line identity mismatch')
        actual_env = dict(raw.decode(errors='replace').split('=', 1) for raw in
                          (proc_root / str(pid) / 'environ').read_bytes().split(b'\0') if b'=' in raw)
        if any(actual_env.get(key) != value for key, value in expected.get('runtime_flags', {}).items()):
            raise ValueError('Controller runtime control identity mismatch')
        attested = file_identity(identity_file)
    path = Path(manifest_file).resolve() if manifest_file else root / 'RELEASE.json'
    if (root / 'RELEASE.json').is_file() and path != root / 'RELEASE.json':
        raise ValueError('Immutable engine requires its own RELEASE.json; override the inherited legacy manifest')
    manifest = json.loads(path.read_text())
    legacy = 'prod_root' in manifest
    if legacy:
        if not manifest_file or Path(manifest['prod_root']).resolve() != root:
            raise ValueError('Legacy native snapshot must be explicitly supplied for the actual engine root')
        kind = 'controller legacy snapshot; mutable tree checked against frozen hashes'
    else:
        if path.name != 'RELEASE.json' or path.parent != root or manifest.get('dirty') is not False:
            raise ValueError('Immutable engine requires its own clean RELEASE.json')
        index = root / 'RELEASE.files.sha256'
        if file_identity(index)['sha256'] != manifest.get('files_sha256'):
            raise ValueError('Frozen release file index hash mismatch')
        kind = 'immutable release'
    expected_so = manifest.get('so')
    if not isinstance(expected_so, dict) or not expected_so:
        raise ValueError('Native manifest has no frozen .so hashes')
    wanted = {}
    basenames = set()
    for relative, digest in expected_so.items():
        candidate = root / relative
        resolved = candidate.resolve()
        if (Path(relative).is_absolute() or '..' in Path(relative).parts or not resolved.is_relative_to(root)
                or not str(relative).endswith('.so') or not isinstance(digest, str)
                or not re.fullmatch(r'[0-9a-f]{64}', digest)):
            raise ValueError('Invalid native manifest entry')
        if file_identity(resolved)['sha256'] != digest:
            raise ValueError(f'Frozen native hash mismatch: {relative}')
        wanted[resolved] = {'relative': relative, 'sha256': digest}
        basenames.add(resolved.name)
    processes = _process_tree(pid, proc_root)
    loaded, workers = [], []
    for current in processes:
        start = _process_start(current, proc_root)
        seen = set()
        for line in (proc_root / str(current) / 'maps').read_text().splitlines():
            fields = line.split(None, 5)
            if len(fields) < 6:
                continue
            mapped = fields[5].replace('\\040', ' ')
            clean = mapped.removesuffix(' (deleted)')
            candidate = Path(clean)
            # The legacy root contains its real .venv. Those third-party torch,
            # numpy and uvloop libraries are not part of snapshot.so. Only own
            # vLLM/JIT native directories and manifested basename collisions
            # belong to this gate; off-tree copies of our kernels still fail.
            own_directory = candidate.is_absolute() and any(candidate.is_relative_to(root / name) for name in ('vllm', '.deps'))
            relevant = (candidate.name in basenames or '/vllm/' in clean or own_directory) and '.so' in candidate.name
            if not relevant:
                continue
            if mapped.endswith(' (deleted)'):
                raise ValueError('Deleted native mapping in engine process')
            resolved = candidate.resolve()
            if resolved not in wanted:
                raise ValueError(f'Off-tree or unmanifested native mapping: {candidate.name}')
            stat = resolved.stat()
            major, minor = (int(part, 16) for part in fields[3].split(':'))
            if stat.st_ino != int(fields[4]) or os.major(stat.st_dev) != major or os.minor(stat.st_dev) != minor:
                raise ValueError(f'Mapped native inode/device was replaced: {candidate.name}')
            if resolved not in seen:
                loaded.append({'pid': current, 'pid_start': start, **wanted[resolved],
                               'inode': stat.st_ino, 'device': fields[3]})
                seen.add(resolved)
        if _process_start(current, proc_root) != start:
            raise ValueError('Engine worker PID/start changed during native attestation')
        workers.append({'pid': current, 'pid_start': start})
    if not loaded:
        raise ValueError('No manifest-owned native mappings found in engine process tree')
    if _process_start(pid, proc_root) != str(identity['start_ticks']):
        raise ValueError('Engine PID/start changed during native attestation')
    return {'verified': True, 'kind': kind, 'root': str(root), 'manifest': file_identity(path),
            'controller_identity_file': attested, 'source_sha': manifest.get('sha'),
            'expected_native_files': len(wanted), 'processes': workers, 'loaded': loaded,
            'scope': 'mapped host ELF files and disk hashes; does not prove CUDA dispatch or launch arguments'}
