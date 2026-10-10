#!/usr/bin/env python3
"""Controlled direct-engine or authenticated local gateway stream measurements.

Use --prepare FILE offline to create synthetic prose and quality-tool requests.
Live gateway runs need a window's 0600 token and sanitized identity files. They
never request remote routes. This measures client arrival, not browser render.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import statistics
import sys
import time
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evalkit'))
import measurement
import scoring

ROOT = Path(__file__).resolve().parents[1]
IDENTITY_KEYS = ('window', 'phase', 'engine_pid', 'root', 'fingerprint')


def tokenize_prompt(base, payload):
    """CPU-only served-template tokenization, without submitting generation."""
    request = {k: payload[k] for k in ('model', 'messages', 'chat_template_kwargs', 'tools') if k in payload}
    request['add_generation_prompt'] = True
    response = measurement.requests.post(base.rstrip('/') + '/tokenize', json=request, timeout=120)
    response.raise_for_status()
    data = response.json()
    count = data.get('count')
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError('Tokenizer returned no valid count')
    return count


def exact_prompt(target, base, settings):
    """Build deterministic varied notes, then fit the actual served template.

    Binary search may cut the synthetic source at a character boundary; the
    complete task remains appended and no server truncation is requested.
    """
    task = '\nWrite a numbered checklist of 70 serving measurements. Explain each in two useful sentences. Continue through the checklist.'
    paragraphs = []
    # Intentionally varied records rather than one repeated paragraph.
    for index in range(max(300, target // 10)):
        paragraphs.append(f'Record {index:06}: cluster {index % 73}, queue {index % 19}, release {index % 97}. '
            f'Validate model roots and tool contracts; retain request receipts. Measurement {index * 17 + 3} '
            'distinguishes prefill, generation, cache state, scheduling and transfer costs. Observe client output arrival and verify tool arguments.\n')
    document = ''.join(paragraphs)
    lo, hi = 0, len(document)
    best = None
    for _ in range(25):
        mid = (lo + hi) // 2
        payload = measurement.merge_request({'model': 'qwen-local'},
            {'messages': [{'role': 'user', 'content': document[:mid] + task}]}, settings)
        count = tokenize_prompt(base, payload)
        if best is None or abs(count-target) < abs(best[1]-target):
            best = (payload, count)
        if abs(count-target) <= max(1, target//1000):
            return payload, count
        if count < target:
            lo = mid + 1
        else:
            hi = mid - 1
        if lo > hi:
            break
    if not best or abs(best[1]-target) > target * .01:
        raise ValueError('Cannot fit requested prompt budget within one percent')
    return best


def prepared_cases(prompt_tokens=32768, tokenize_base=None, profile_names=('greedy',)):
    cases = []
    for profile in profile_names:
        settings = measurement.load_profile(ROOT / 'evalkit' / 'profiles' / (profile + '.json'))
        if tokenize_base:
            request, actual = exact_prompt(prompt_tokens, tokenize_base, settings)
        else:
            # Unit tests and offline plans can describe an unfitted case, but
            # the live runner refuses it. No characters-to-tokens claim.
            request = measurement.merge_request({'model': 'qwen-local'},
                {'messages': [{'role': 'user', 'content': 'OFFLINE PLACEHOLDER: tokenize before any live run.'}]}, settings)
            actual = None
        for output in (512, 1024):
            cases.append({'id': f'prose_{prompt_tokens}_{output}_{profile}', 'kind': 'prose',
                          'requested_prompt_tokens': prompt_tokens, 'served_prompt_tokens': actual,
                          'request': {**request, 'max_tokens': output}})
        fixture = json.loads((ROOT / 'evalkit/items/tool_call/item_001.json').read_text())
        tool_request = measurement.merge_request({'model': 'qwen-local'}, fixture['request'], settings)
        cases.append({'id': f'tool_{profile}', 'kind': 'tool', 'fixture': fixture,
                      'served_prompt_tokens': tokenize_prompt(tokenize_base, tool_request) if tokenize_base else None,
                      'request': tool_request})
    return {'schema': 1, 'origin': 'vLLM 2080 Ti Definitive Edition; github.com/weicj; upstream vLLM',
            'profiles_note': 'greedy control; optional current gateway absent-field nonthinking defaults; active Halo overrides separately recorded',
            'context_note': 'Exact served-template /tokenize count; live usage must match within one percent',
            'tokenize_base': tokenize_base, 'cases': cases, 'suite_sha256': measurement.canonical_hash(cases)}


def window_credentials(token_file, identity_file):
    if not token_file or not identity_file:
        raise ValueError('Gateway measurements require window token and identity files')
    path = Path(token_file)
    stat = path.stat()
    if stat.st_mode & 0o077 or stat.st_uid != os.getuid():
        raise ValueError('Window token must be owned by the current user with mode 0600 or stricter')
    token = path.read_text().strip()
    if not token:
        raise ValueError('Empty window token')
    data = json.loads(Path(identity_file).read_text())
    expected = {key: data.get(key) for key in IDENTITY_KEYS}
    if any(value is None for value in expected.values()):
        raise ValueError('Incomplete sanitized window identity')
    return {'X-Window-Token': token, 'X-Client': 'window-measurement-local-pin'}, expected


def verify_receipt(result, expected):
    headers = {k.lower(): v for k, v in result.get('route_receipts', {}).items()}
    if headers.get('x-gateway-route') != 'local':
        return 'missing local route receipt'
    try:
        actual = json.loads(headers.get('x-window-identity', ''))
    except (TypeError, ValueError):
        return 'missing/malformed window identity receipt'
    if any(actual.get(key) != value for key, value in expected.items()):
        return 'window/runtime identity receipt mismatch'
    return None


def measure_case(base, case, index, headers=None, expected=None, nonce=None):
    payload = dict(case['request'])
    if expected:
        payload['model'] = 'estate-local'
    # Cold-prefix nonces are deterministic and identical across baseline and
    # candidate arms. Warm requests repeat the complete per-stream payload.
    if case['kind'] == 'prose':
        payload['messages'] = [dict(m) for m in payload['messages']]
        if nonce:
            payload['messages'][0]['content'] = f'Measurement nonce {nonce}:{index}.\n' + payload['messages'][0]['content']
        else:
            payload['messages'][-1]['content'] += f'\nMeasurement stream {index}.'
    result = measurement.post_chat(base, payload, timeout_s=900, headers=headers)
    error = verify_receipt(result, expected) if expected and result['status'] == 'ok' else None
    quality = None
    if result['status'] == 'ok' and case['kind'] == 'tool':
        message = result['response']['choices'][0]['message']
        quality = scoring.score_tool_call(case['fixture'], message)
        if not quality['passed']:
            error = 'tool name/argument contract failed'
    timeline = result.get('timeline') or {}
    if result['status'] == 'ok' and case['kind'] == 'prose' and timeline.get('first_prose_s') is None:
        error = 'no visible prose generated'
    usage = result.get('response', {}).get('usage') or {}
    tokens = usage.get('completion_tokens')
    requested = case.get('served_prompt_tokens')
    actual_prompt = usage.get('prompt_tokens')
    if result['status'] == 'ok' and requested is not None and (not isinstance(actual_prompt, int) or abs(actual_prompt-requested) > max(8, requested*.01)):
        error = 'generation prompt_tokens differ from served-template preparation by more than one percent'
    arrivals = timeline.get('generated_event_times_s') or []
    decode = arrivals[-1]-arrivals[0] if len(arrivals) > 1 else None
    return {'case': case['id'], 'stream': index, 'status': result['status'],
            'valid': result['status'] == 'ok' and not error, 'error': error or result.get('error'),
            'timeline': timeline, 'usage': usage, 'prepared_prompt_tokens': requested,
            'request_sha256': result.get('request_payload_sha256'),
            'counted_generated_tokens_s': timeline.get('counted_generated_tokens_s'),
            'sampling': {k: v for k, v in payload.items() if k in measurement.SAMPLING_KEYS},
            'estimated_decode_tokens_s': (tokens-1)/decode if tokens and decode else None,
            'rate_semantics': 'first-to-last generated SSE delta; unknown token count in first bundle',
            'tool_contract_passed': quality['passed'] if quality else None,
            'first_correct_tool_contract_s': timeline.get('all_tools_complete_s') if quality and quality['passed'] else None,
            'tool_effect': 'not executed; mechanically scored synthetic contract' if quality else None,
            'speculative_decoding': result.get('response', {}).get('metrics', {}).get('speculative_decoding'),
            'route_receipts': result.get('route_receipts', {})}


def wave_summary(rows, concurrency, wall):
    valid = [r for r in rows if r['valid']]
    def values(key):
        return [r['timeline'][key] for r in valid if r['timeline'].get(key) is not None]
    rates = [r['estimated_decode_tokens_s'] for r in valid if r['estimated_decode_tokens_s'] is not None]
    counted_rates = [r['counted_generated_tokens_s'] for r in valid if r.get('counted_generated_tokens_s') is not None]
    useful = values('first_useful_candidate_s')
    gaps = values('generated_gap_max_s')
    return {'concurrency': concurrency, 'valid_requests': len(valid), 'total_requests': len(rows),
            'wave_wall_s': wall, 'median_first_useful_candidate_s': statistics.median(useful) if useful else None,
            'median_counted_per_stream_generated_tokens_s': statistics.median(counted_rates) if counted_rates else None,
            'slowest_counted_per_stream_generated_tokens_s': min(counted_rates) if counted_rates else None,
            'median_estimated_per_stream_decode_tokens_s': statistics.median(rates) if rates else None,
            'slowest_estimated_per_stream_decode_tokens_s': min(rates) if rates else None,
            'aggregate_end_to_end_tokens_s': sum(r.get('usage', {}).get('completion_tokens', 0) for r in valid)/wall if wall else None,
            'worst_generated_event_gap_s': max(gaps) if gaps else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', type=Path, help='prepare synthetic requests; --tokenize-base fits CPU-only served template')
    parser.add_argument('--prompt-tokens', type=int, default=32768)
    parser.add_argument('--tokenize-base', help='loopback engine URL for CPU-only /tokenize, never generation')
    parser.add_argument('--profiles', default='greedy')
    parser.add_argument('--suite', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--base', default='http://127.0.0.1:8001')
    parser.add_argument('--engine-pid', type=int, default=int(os.environ.get('WINDOW_ENGINE_PID', '0')) or None)
    parser.add_argument('--concurrency', default='1,2,4,8')
    parser.add_argument('--reps', type=int, default=3)
    parser.add_argument('--cache-regime', choices=('shared-prefix', 'unique-prefix'), default='shared-prefix')
    parser.add_argument('--warmup', type=int, choices=(0, 1), default=1, help='excluded short prose warmup')
    parser.add_argument('--case', help='comma separated prepared case IDs')
    parser.add_argument('--token-file', default=os.environ.get('WINDOW_BENCH_TOKEN_FILE'))
    parser.add_argument('--window-identity', default=os.environ.get('WINDOW_BENCH_IDENTITY_FILE'))
    args = parser.parse_args(argv)
    if args.prepare:
        args.prepare.parent.mkdir(parents=True, exist_ok=True)
        if args.tokenize_base not in (None, 'http://127.0.0.1:8001', 'http://localhost:8001'):
            parser.error('Tokenization uses only the loopback engine :8001')
        if not 1024 <= args.prompt_tokens <= 262144:
            parser.error('Requested prompt budget must be 1024..262144')
        profiles = tuple(args.profiles.split(','))
        if not profiles or set(profiles) - {'greedy', 'halo-non-thinking'}:
            parser.error('Use shipped greedy and/or halo-non-thinking profiles')
        preparation_identity = measurement.runtime_identity(args.engine_pid) if args.engine_pid else None
        if args.tokenize_base and not preparation_identity:
            parser.error('Served tokenization preparation requires --engine-pid for provenance')
        if preparation_identity:
            measurement.verify_models_root(args.tokenize_base or 'http://127.0.0.1:8001', preparation_identity)
        suite = prepared_cases(args.prompt_tokens, args.tokenize_base, profiles)
        if preparation_identity:
            if measurement.runtime_identity(args.engine_pid)['fingerprint'] != preparation_identity['fingerprint']:
                parser.error('Engine identity changed during tokenization preparation')
            suite['tokenizer_provenance'] = preparation_identity
        args.prepare.write_text(json.dumps(suite, indent=2) + '\n')
        print(json.dumps({'prepared_cases': len(suite['cases']), 'suite_sha256': suite['suite_sha256']}))
        return 0
    if not args.engine_pid and args.window_identity:
        candidate = json.loads(Path(args.window_identity).read_text()).get('engine_pid')
        if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
            args.engine_pid = candidate
    url = urlparse(args.base)
    if url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost') or url.port not in (8000, 8001) or url.path not in ('', '/') or url.username or url.password:
        parser.error('Use the loopback gateway :8000 or engine :8001 without URL credentials')
    if not args.suite or not args.out or not args.engine_pid:
        parser.error('Live runs require --suite --out --engine-pid')
    counts = [int(x) for x in args.concurrency.split(',')]
    if not counts or any(n < 1 or n > 16 for n in counts) or not 1 <= args.reps <= 20:
        parser.error('Use concurrency 1..16 and reps 1..20')
    suite = json.loads(args.suite.read_text())
    if suite.get('suite_sha256') != measurement.canonical_hash(suite['cases']):
        parser.error('Prepared suite hash mismatch')
    cases = suite['cases']
    if any(c.get('served_prompt_tokens') is None for c in cases):
        parser.error('Suite must be fitted through served /tokenize before any live run')
    if args.case:
        selected = set(args.case.split(','))
        cases = [c for c in cases if c['id'] in selected]
        if selected != {c['id'] for c in cases}:
            parser.error('Unknown case ID')
    headers = expected = None
    if url.port == 8000:
        headers, expected = window_credentials(args.token_file, args.window_identity)
    provenance = measurement.runtime_identity(args.engine_pid)
    provenance['model_roots'] = measurement.verify_models_root('http://127.0.0.1:8001', provenance)
    measurement.require_offline_window(drained=True)
    provenance['measurement_files'] = [measurement.file_identity(Path(__file__)), measurement.file_identity(ROOT/'evalkit/measurement.py')]
    if expected and (expected['engine_pid'] != args.engine_pid or expected['root'] != provenance['source']['path']):
        parser.error('Window identity does not match actual engine PID/source')
    output = {'schema': 1, 'started_unix': time.time(), 'base': args.base,
              'suite_sha256': suite['suite_sha256'], 'provenance': provenance,
              'window_identity': expected, 'conditions': {'browser_rendering': 'not measured',
                  'background_isolation': 'window credential' if expected else 'controller must attest engine exclusivity',
                  'cache_regime': args.cache_regime, 'warmup': 'excluded' if args.warmup else 'none'},
              'waves': [], 'requests': []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        parser.error('Use a fresh output path; measurement results are not overwritten')
    if args.warmup:
        warm = {'id': 'excluded_kernel_warmup', 'kind': 'prose', 'request': {
            'model': 'qwen-local', 'messages': [{'role': 'user', 'content': 'Explain inference scheduling in several sentences.'}],
            'max_tokens': 32, 'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}}}
        output['warmup'] = measure_case(args.base, warm, 0, headers, expected)
        if not output['warmup']['valid']:
            output['valid_run'] = False
            args.out.write_text(json.dumps(output, indent=2) + '\n')
            return 1
    for case in cases:
        for count in counts:
            for rep in range(args.reps):
                measurement.require_offline_window(drained=True)
                nonce = f"{case['id']}:{count}:{rep}" if args.cache_regime == 'unique-prefix' else None
                if args.cache_regime == 'shared-prefix' and case['kind'] == 'prose':
                    # Prime the exact matched fixture independently for every
                    # wave. Priming output is excluded; cached token usage is
                    # retained in the measured requests, never assumed.
                    prime = {**case, 'request': {**case['request'], 'max_tokens': 16}}
                    primes = [measure_case(args.base, prime, i, headers, expected) for i in range(count)]
                    output.setdefault('cache_priming', []).append({'case': case['id'], 'rep': rep, 'concurrency': count, 'requests': primes})
                    if not all(r['valid'] for r in primes):
                        output['valid_run'] = False
                        args.out.write_text(json.dumps(output, indent=2) + '\n')
                        return 1
                before = measurement.scrape_spec_counters('http://127.0.0.1:8001')
                start = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
                    rows = list(pool.map(lambda i: measure_case(args.base, case, i, headers, expected, nonce), range(count)))
                wall = time.perf_counter()-start
                after = measurement.scrape_spec_counters('http://127.0.0.1:8001')
                summary = {**wave_summary(rows, count, wall), 'case': case['id'], 'rep': rep,
                           'engine_spec_counter_delta': measurement.spec_counter_delta(before, after, isolated=bool(expected))}
                output['requests'].extend([{**r, 'rep': rep, 'concurrency': count} for r in rows])
                output['waves'].append(summary)
                # Persist completed waves on failure; no request bodies/text/tokens.
                try:
                    output['runtime_unchanged'] = measurement.runtime_identity(args.engine_pid)['fingerprint'] == provenance['fingerprint']
                except (OSError, ValueError):
                    output['runtime_unchanged'] = False
                output['valid_run'] = output['runtime_unchanged'] and all(r['valid'] for r in output['requests'])
                args.out.write_text(json.dumps(output, indent=2) + '\n')
                print(json.dumps(summary), flush=True)
                if not output['valid_run']:
                    return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
