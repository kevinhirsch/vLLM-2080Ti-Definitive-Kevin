# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-only SM75 continuation qualifier; description is the default.

Based on upstream vLLM and weicj/vLLM-2080Ti-Definitive b7ea65b3ee.
No Torch/vLLM imports, HTTP, service observations, JIT or GPU calls occur in
description mode. --run-gpu is solely for a root-owned exclusive window.
The existing immutable FlashQLA ELF is loaded directly, never source-built.
Triton ops may JIT during an authorized future GPU run. This is numerical
qualification, not a serving/timing harness or a full-production claim.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS

HERE = Path(__file__).resolve().parent
BINDING = HERE / 'flashqla_subblock_source_binding.json'
SOURCE_COMMIT = 'b7ea65b3ee8a60bcb0b7f1a256c0915badc77697'
DEFAULT_ROOT = Path('/home/kevin/.local/share/vllm-releases/halo-prefill-global-quota-20261010-b7ea65b3ee-20261010T1943')
MODEL = Path('/home/kevin/Desktop/models/Qwen3.8-27B-HauhauCS-Aggressive-W4A16-twolven')
QUIET = Path('/home/kevin/Desktop/frontier-lab/qualification/global-quota-mtp3-quiet/quiet-state-entrypoint.py')
MEASUREMENT = Path('/home/kevin/Desktop/wt-frontier-measurement/deploy/evalkit/measurement.py')
GDN_REL = '.deps/FlashQLA-SM70-SM75/.torch_extensions_vllm_flashqla_legacy/flash_qla_legacy_gdn/flash_qla_legacy_gdn.so'
LENGTHS = (511, 512, 513, 1023, 1024, 1025, 3567, 3568, 3569, 7135, 7136, 7137)
BLOCK, HASH, HK, HV, D, CONV_DIM, HISTORY = 3568, 16, 8, 24, 128, 5120, 6
TOLERANCES = {'conv': (4e-3, 2e-3), 'prepared': (2e-3, 2e-3),
              'output': (2e-3, 2e-3), 'state': (2e-5, 2e-3)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def chunk_ends(length, threshold, *, partial=True):
    """Independent plan from boundary contracts, not an imported scheduler.

    Physical boundaries and the Eagle-drop partial-tail checkpoint are stops.
    An intermediate sub-block state remains private. hash16 is this fixture's
    explicit source contract; actual worker APC publication is not simulated.
    """
    require(length in LENGTHS and threshold in (512, 1024), 'bounded matrix required')
    tail = max(length // HASH * HASH - HASH, 0)
    last_cache = max(length // BLOCK * BLOCK - BLOCK, 0)
    stops = {length, *(range(BLOCK, length, BLOCK))}
    if partial and last_cache < tail < length:
        stops.add(tail)
    position, ends = 0, []
    while position < length:
        position = min(position + threshold,
                       min(stop for stop in stops if stop > position))
        ends.append(position)
    return ends


def vk_discriminator():
    """Non-symmetric state distinguishes VK from the old vendor KV oracle."""
    h, k = ((1., 2.), (3., 5.)), (.25, -.5)
    vk = [sum(h[v][i] * k[i] for i in range(2)) for v in range(2)]
    kv = [sum(h[i][v] * k[i] for i in range(2)) for v in range(2)]
    return {'state': 'non-symmetric synthetic2x2', 'VK_state_times_k': vk,
            'old_KV_transpose_times_k': kv, 'different': vk != kv}


def prepare(root, model, lengths):
    binding = json.loads(BINDING.read_text())
    require(binding['source_commit'] == SOURCE_COMMIT, 'binding commit changed')
    require(all(length in LENGTHS for length in lengths) and lengths
            and len(set(lengths)) == len(lengths), 'unique bounded lengths required')
    actual = {name: sha(root / name) for name in binding['source_sha256']}
    require(actual == binding['source_sha256'], 'numerical source differs from pinned b7ea')
    require(sha(model / 'config.json') == binding['checkpoint_config_sha256'], 'checkpoint changed')
    c = json.loads((model / 'config.json').read_text())['text_config']
    geometry = (c['linear_num_key_heads'] // 2, c['linear_num_value_heads'] // 2,
                c['linear_key_head_dim'], c['linear_value_head_dim'],
                c['linear_conv_kernel_dim'], c['mamba_ssm_dtype'])
    require(geometry == (8, 24, 128, 128, 4, 'float32'), 'production geometry changed')
    return {'schema': 'flashqla-subblock-plan-v1', 'source_commit': SOURCE_COMMIT,
            'source_root': str(root.resolve()), 'source_sha256': actual,
            'binding_sha256': sha(BINDING), 'entrypoint_sha256': sha(__file__),
            'checkpoint_config_sha256': sha(model / 'config.json'),
            'lengths': list(lengths), 'thresholds': [512, 1024],
            'split_positions': {str(n): {str(t): chunk_ends(n, t) for t in (512, 1024)} for n in lengths},
            'local_tp2': {'key_heads': HK, 'value_heads': HV, 'key_dim': D, 'value_dim': D,
                          'conv_dim': CONV_DIM, 'conv_width': 4, 'conv_history_plus_drafts': HISTORY},
            'physical_block_fixture': BLOCK, 'hash_block_fixture': HASH,
            'mtp3_query_width': 4, 'accepted_prefixes': [1, 2, 3, 4],
            'state_storage': 'VK [value,key], float32; native h0 aliases final_state',
            'normalization': 'x/sqrt(sum(x*x)+1e-6), quantized to FP16 after preparation',
            'tolerances_atol_rtol': TOLERANCES, 'vk_cpu_discriminator': vk_discriminator(),
            'gpu_executed': False, 'full_production_state_qualified': False,
            'allocator_apc_cow_qualified': False, 'logit_trajectory_qualified': False,
            'limits': ['synthetic one-linear-layer inputs, not production weights/model logits',
                       'private fork exercises real pre-copy/conv/native varlen ops, not allocator refcounts/APC admission',
                       'whole native prefix is numerical reference, not a legal whole-model scheduling arm',
                       'numerical CPU transfers and Triton JIT may occur only in explicit future GPU execution',
                       'quiet samples do not prevent future direct traffic; controller renewal/deadman/paused consumers required']}


def load(path, digest, name):
    require(sha(path) == digest, 'admission helper changed; re-review required')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def cache_dtypes(raw_argv):
    """Read only these known nonsecret argv controls; never save full argv.

    The checkpoint's mamba_ssm_dtype metadata does not override vLLM's
    cache_config: auto SSM follows the conv/model dtype on this source.
    """
    args = raw_argv.decode().rstrip('\0').split('\0')
    found = {}
    for i, arg in enumerate(args):
        key, equal, value = arg.partition('=')
        if key not in ('--mamba-cache-dtype', '--mamba-ssm-cache-dtype'):
            continue
        require(key not in found, 'duplicate cache dtype control')
        if not equal:
            require(i+1 < len(args) and not args[i+1].startswith('--'), 'missing cache dtype value')
            value = args[i+1]
        found[key] = value
    conv = found.get('--mamba-cache-dtype', 'auto')
    ssm = found.get('--mamba-ssm-cache-dtype', 'auto')
    require(conv in ('auto', 'float16'), 'actual conv cache must be FP16')
    require(ssm == 'float32', 'actual SSM cache must explicitly be FP32')
    return {'conv': 'float16', 'ssm': 'float32'}


def admit(args, plan):
    """Everything here precedes Torch import; uses the separately audited gate."""
    binding = json.loads(BINDING.read_text())
    quiet = load(QUIET, binding['quiet_helper_sha256'], 'subblock_quiet_gate')
    m = load(MEASUREMENT, binding['measurement_sha256'], 'subblock_measurement')
    manifest = args.source_root / 'RELEASE.json'
    gate_args = NS(source_root=args.source_root, source_commit=SOURCE_COMMIT,
                   native_manifest=str(manifest), engine_unit='vllm-qwen27b.service',
                   wait_s=30, sample_gap_s=.5, minimum_fence_s=30,
                   engine='http://127.0.0.1:8001', gateway='http://127.0.0.1:8000',
                   out=args.out, numerical_out=args.out.with_suffix('.unused.json'))
    before = quiet.quiet_gate(m, gate_args)
    identity = m.runtime_identity(before['engine_pid'])
    require(identity['fingerprint'] == before['engine_identity_fingerprint'], 'engine changed after quiet admission')
    require(Path(identity['checkpoint']['root']).resolve() == args.model.resolve(), 'actual checkpoint differs')
    require(identity['flags'].get('--dtype') in ('float16', 'half'), 'actual model dtype must be FP16')
    require(identity['flags'].get('--tensor-parallel-size') == '2', 'actual TP2 required')
    dtypes = cache_dtypes((Path('/proc') / str(before['engine_pid']) / 'cmdline').read_bytes())
    before = {**before, 'actual_cache_dtypes': dtypes}
    spec = json.loads(identity['flags'].get('--speculative-config', '{}'))
    require(spec.get('num_speculative_tokens') == 3, 'actual runtime must use production MTP3')
    additional = json.loads(identity['flags'].get('--additional-config', '{}'))
    # EngineArgs explicitly folds the CLI selector into additional_config,
    # overwriting any earlier value; /proc still shows the original inputs.
    backend = identity['flags'].get('--gdn-prefill-backend', additional.get('gdn_prefill_backend'))
    require(backend == 'flashqla_legacy', 'actual backend must be FlashQLA legacy')
    native = m.verify_loaded_native(identity, os.environ['WINDOW_ENGINE_IDENTITY_FILE'], str(manifest))
    extension = args.source_root / GDN_REL
    require(sha(extension) == binding['gdn_elf_sha256'], 'existing native GDN ELF changed')
    require(any(row['relative'] == GDN_REL and row['sha256'] == binding['gdn_elf_sha256']
                for row in native['loaded']), 'required GDN ELF is not mapped by attested engine')
    require(prepare(args.source_root, args.model, plan['lengths']) == plan, 'source changed before numerical entry')
    # The immutable receipt does not expose this host env by default. Read only
    # the one known parameter; do not record arbitrary argv/environ or secrets.
    name = b'VLLM_FLASHQLA_LEGACY_GDN_QKV_DTYPE='
    values = [item[len(name):].decode('ascii') for item in
              (Path('/proc') / str(before['engine_pid']) / 'environ').read_bytes().split(b'\0')
              if item.startswith(name)]
    require(len(values) <= 1, 'ambiguous actual QKV dtype mode')
    mode = values[0] if values else 'native'
    require(mode in ('native', 'fp32'), 'unsupported actual QKV dtype mode')
    return quiet, m, gate_args, before, mode, extension


def compare(torch, actual, expected, name, rows, *, kind='state', exact=False, **metadata):
    """Every element compared; no fraction-of-elements warning exemption."""
    a, b = actual.detach().cpu(), expected.detach().cpu()
    require(torch.isfinite(a).all().item() and torch.isfinite(b).all().item(), 'nonfinite ' + name)
    atol, rtol = (0, 0) if exact else TOLERANCES[kind]
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    delta = (a.double() - b.double()).abs()
    rows.append({'test': name, 'passed': True, 'elements': a.numel(), 'dtype': str(a.dtype),
                 'all_elements_finite': True, 'max_abs_error': delta.max().item() if delta.numel() else 0,
                 'atol': atol, 'rtol': rtol, **metadata})


def conv_oracle(torch, raw, weight, bias, initial):
    # Independent CPU FP64 depthwise convolution, then FP16 output/state.
    x = torch.cat((initial[:, :3].double(), raw.double().T), dim=1)
    acc = bias.double()[:, None].expand(CONV_DIM, raw.shape[0]).clone()
    for tap in range(4):
        acc += x[:, tap:tap + raw.shape[0]] * weight[:, tap:tap + 1].double()
    return (acc.sigmoid() * acc).T.half(), x[:, -3:].half()


def prep_oracle(torch, mixed, a, b, a_log, dt):
    q, k, v = torch.split(mixed.double(), (HK * D, HK * D, HV * D), -1)
    q, k = q.view(-1, HK, D), k.view(-1, HK, D)
    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    gate = -a_log.double().exp() * torch.nn.functional.softplus(a.double() + dt.double())
    return q.half(), k.half(), v.view(-1, HV, D).half(), gate.float(), b.double().sigmoid().float()


def delta_oracle(torch, prepared, initial, checkpoints):
    """Independent CPU VK recurrence; retain only required checkpoint states."""
    q, k, v, gate, beta = (x.detach().cpu().double() for x in prepared)
    state = initial.detach().cpu().double().clone()
    states, outputs = {}, []
    ratio = HV // HK
    for token in range(q.shape[0]):
        qt = q[token].repeat_interleave(ratio, 0) * D ** -.5
        kt = k[token].repeat_interleave(ratio, 0)
        decay = gate[token].exp()[:, None, None]
        decayed = state * decay
        residual = (v[token] - torch.einsum('hvk,hk->hv', decayed, kt)) * beta[token, :, None]
        state = (decayed + residual[:, :, None] * kt[:, None, :]).float().double()
        outputs.append(torch.einsum('hvk,hk->hv', state, qt).half())
        if token + 1 in checkpoints:
            states[token + 1] = state.float().clone()
    return torch.stack(outputs), states


def split_qkv(mixed):
    q, k, v = mixed.split((HK * D, HK * D, HV * D), dim=-1)
    return q.view(1, -1, HK, D), k.view(1, -1, HK, D), v.view(1, -1, HV, D)


def native_call(torch, ext, prepared, state, mode, *, offsets=None):
    dtype = torch.float32 if mode == 'fp32' else torch.float16
    q, k, v, gate, beta = prepared
    values = [x.to(dtype).contiguous().unsqueeze(0) for x in (q, k, v)]
    values += [gate.float().contiguous().unsqueeze(0), beta.float().contiguous().unsqueeze(0)]
    if offsets is None:
        output, final = ext.gdn_forward(*values, state, D ** -.5)
    else:
        output, final = ext.gdn_forward_varlen(*values, state, D ** -.5, offsets)
    require(final.data_ptr() == state.data_ptr(), 'native supplied-state alias contract changed')
    return output.squeeze(0).half(), final


def prefix(torch, ext, ops, raw, a, b, weights, bias, a_log, dt, conv0, ssm0,
           ends, mode, rows, expected_conv, expected, expected_states, *, packed=True):
    conv = conv0.cuda().unsqueeze(0).clone()
    state = ssm0.cuda().unsqueeze(0).clone()
    outputs, start = [], 0
    for end in ends:
        x = raw[start:end].cuda().contiguous()
        mixed = ops.conv_fn(x.T, weights.cuda(), bias.cuda(), conv,
                            torch.tensor([0, end-start], dtype=torch.int32, device='cuda'),
                            cache_indices=torch.tensor([0], dtype=torch.int32, device='cuda'),
                            has_initial_state=torch.tensor([True], device='cuda'),
                            activation='silu', null_block_id=-1).T
        prepared = ops.prep(mixed.contiguous(), a[start:end].cuda(), b[start:end].cuda(),
                            a_log.cuda(), dt.cuda(), HK, D, D, True, False)
        compare(torch, mixed, expected_conv[start:end], 'prefix conv outputs', rows,
                kind='conv', query_start=start, query_end=end)
        reference_prep = prep_oracle(torch, mixed.cpu(), a[start:end], b[start:end], a_log, dt)
        for name, actual, reference in zip(('q', 'k', 'v', 'log_gate', 'beta'), prepared, reference_prep):
            compare(torch, actual, reference, 'post-conv preparation ' + name, rows,
                    kind='prepared', query_start=start, query_end=end)
        offsets = torch.tensor([0, end-start], dtype=torch.int32, device='cuda') if packed else None
        out, state = native_call(torch, ext, prepared, state, mode, offsets=offsets)
        outputs.append(out.cpu())
        compare(torch, state[0], expected_states[end], 'prefix-state trajectory', rows,
                query_start=start, query_end=end, logical_state_position=end,
                cache_role='full-block checkpoint' if end % BLOCK == 0 else 'private/tail state; publication not tested')
        start = end
    compare(torch, torch.cat(outputs), expected[:ends[-1]], 'conv-to-FlashQLA outputs', rows, kind='output')
    joined = torch.cat((conv0[:, :3], raw[:ends[-1]].T), dim=1)
    compare(torch, conv[0, :, :3], joined[:, -3:], 'prefill conv history', rows, exact=True)
    return conv[0], state[0]


def verification(torch, ops, conv, ssm, data, weights, bias, a_log, dt, rows, label):
    """Real first MTP4 query, then accepted1..4 rollback and next MTP4 query."""
    raw, a, b = data
    cp = torch.stack((conv, conv, conv, conv, conv, conv)).clone()
    hp = torch.stack((ssm, ssm, ssm, ssm, ssm, ssm)).clone()
    indices = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32, device='cuda')
    cu = torch.tensor([0, 4], dtype=torch.int32, device='cuda')

    def step(cpool, hpool, start, accepted):
        mixed = ops.conv_update(raw[start:start+4].cuda(), cpool, weights.cuda(), bias.cuda(), 'silu',
                 conv_state_indices=torch.tensor([1], dtype=torch.int32, device='cuda'),
                 num_accepted_tokens=torch.tensor([accepted], dtype=torch.int32, device='cuda'),
                 query_start_loc=cu, max_query_len=4, null_block_id=-1)
        q, k, v = split_qkv(mixed)
        out, _ = ops.spec(A_log=a_log.cuda(), a=a[start:start+4].cuda(), b=b[start:start+4].cuda(),
                 dt_bias=dt.cuda(), q=q, k=k, v=v, initial_state=hpool, inplace_final_state=True,
                 ssm_state_indices=indices, cu_seqlens=cu,
                 num_accepted_tokens=torch.tensor([accepted], dtype=torch.int32, device='cuda'),
                 use_qk_l2norm_in_kernel=True, null_block_id=-1)
        return mixed, out.squeeze(0)

    first_mixed, first_out = step(cp, hp, 0, 1)
    first_ref, _ = conv_oracle(torch, raw[:4], weights, bias, conv.cpu())
    compare(torch, first_mixed, first_ref, 'first MTP conv', rows, kind='conv', arm=label)
    # Spec normalizes raw FP16 q/k in FP32, without the prefill prep's FP16
    # normalized-q/k store. Reproduce that distinct declared arithmetic.
    def spec_reference(mixed, av, bv, state):
        prepared = list(prep_oracle(torch, mixed.cpu(), av, bv, a_log, dt))
        q, k, _ = split_qkv(mixed.cpu())
        prepared[0] = q[0].double() / (q[0].double().square().sum(-1, keepdim=True) + 1e-6).sqrt()
        prepared[1] = k[0].double() / (k[0].double().square().sum(-1, keepdim=True) + 1e-6).sqrt()
        return delta_oracle(torch, prepared, state, {1, 2, 3, 4})
    first_expected, first_states = spec_reference(first_ref, a[:4], b[:4], ssm)
    compare(torch, first_out, first_expected, 'first MTP output', rows, kind='output', arm=label)
    for i in range(4):
        compare(torch, hp[i+1], first_states[i+1], 'first MTP state', rows, arm=label, prefix=i+1)
    compare(torch, cp[[0,2,3,4,5]], torch.stack((conv, conv, conv, conv, conv)),
            'first MTP unrelated conv untouched', rows, exact=True, arm=label)
    compare(torch, hp[[0,5]], torch.stack((ssm, ssm)),
            'first MTP unrelated SSM untouched', rows, exact=True, arm=label)
    for accepted in (1, 2, 3, 4):
        c, h = cp.clone(), hp.clone()
        c_before, h_before = c.clone(), h.clone()
        mixed, out = step(c, h, 4, accepted)
        history = cp[1, :, accepted-1:accepted+2].cpu()
        expected_mixed, _ = conv_oracle(torch, raw[4:8], weights, bias, history)
        compare(torch, mixed, expected_mixed, 'accepted-prefix conv continuation', rows,
                kind='conv', arm=label, accepted=accepted)
        expected_out, states = spec_reference(expected_mixed, a[4:8], b[4:8], first_states[accepted])
        compare(torch, out, expected_out, 'accepted-prefix GDN continuation', rows,
                kind='output', arm=label, accepted=accepted)
        for i in range(4):
            compare(torch, h[i+1], states[i+1], 'accepted-prefix GDN state', rows,
                    arm=label, accepted=accepted, next_prefix=i+1)
        expected_history = torch.cat((cp[1, :, accepted:accepted+2].cpu(), raw[4:8].T), dim=1)
        compare(torch, c[1], expected_history, 'accepted-prefix full conv record', rows, exact=True,
                arm=label, accepted=accepted)
        compare(torch, c[[0,2,3,4,5]], c_before[[0,2,3,4,5]], 'unrelated conv untouched', rows, exact=True)
        compare(torch, h[[0,5]], h_before[[0,5]], 'unrelated SSM untouched', rows, exact=True)


def private_fork(torch, ext, ops, root, checkpoint_conv, checkpoint_ssm,
                 weights, bias, a_log, dt, mode, rows, random):
    """Actual pre-copy and dense/varlen ops with an explicitly assigned fork.

    This does not call the KV allocator or claim a real APC hit. Shared slot1
    represents state@3568; slots2/3 are assigned private continuations. The
    held sibling and shared checkpoint must remain byte-identical after A.
    """
    worker = importlib.import_module('vllm.v1.worker.mamba_utils')
    require(Path(worker.__file__).resolve().is_relative_to(root.resolve()), 'off-root pre-copy op')
    fixture = root / 'tests/kernels/mamba/test_precopy_mamba_align.py'
    tree = ast.parse(fixture.read_text())
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in ('_reference', '_build_meta')]
    require(len(funcs) == 2, 'existing pre-copy reference changed')
    namespace = {'torch': torch, 'NUM_LAYERS': 1, 'CONV_WIDTH': HISTORY}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(fixture), 'exec'), namespace)
    conv = torch.stack([checkpoint_conv, checkpoint_conv,
                       *[random(CONV_DIM, HISTORY).cuda() for _ in range(6)]])
    state = torch.stack([checkpoint_ssm, checkpoint_ssm,
                        *[random(HV, D, D).float().cuda() for _ in range(6)]])
    bt = torch.tensor([[2, 1, 4, 5], [3, 1, 6, 7]], dtype=torch.int32, device='cuda')
    src = torch.ones(2, dtype=torch.int32, device='cuda')
    dst = torch.zeros(2, dtype=torch.int32, device='cuda')
    bias0 = torch.zeros(2, dtype=torch.int32, device='cuda')
    expected_c, expected_h = namespace['_reference']([conv.cpu()], [state.cpu()],
                         bt.cpu(), src.cpu(), dst.cpu(), bias0.cpu(), 2, True)
    meta = namespace['_build_meta']([conv], [state], 'cuda', True)
    bt_ptr = torch.tensor([bt.data_ptr()], dtype=torch.int64, device='cuda')
    worker.precopy_mamba_align_fused_kernel[(2, 2, 1)](dst, src, bias0, bt_ptr, 4, *meta,
                None, 2, COPY_BLOCK_SIZE=1024, CONV_STATE_DIM_FIRST=True,
                HAS_IDX_MAPPING=False, TEMPORAL_TILES=1)
    compare(torch, conv, expected_c[0], 'actual TP2 fork pre-copy conv', rows, exact=True)
    compare(torch, state, expected_h[0], 'actual TP2 fork pre-copy SSM', rows, exact=True)
    c_before, h_before = conv.clone(), state.clone()
    raw_a, raw_b = random(17, CONV_DIM) * .1, random(31, CONV_DIM) * .1
    av_a, av_b = random(17, HV) * .1, random(31, HV) * .1
    bv_a, bv_b = random(17, HV) * .1, random(31, HV) * .1

    def forward(raw, av, bv, slots, lengths):
        offsets = torch.tensor([0, *lengths], dtype=torch.int32, device='cuda')
        # lengths are cumulative ends, not individual sequence lengths.
        mixed = ops.conv_fn(raw.cuda().contiguous().T, weights.cuda(), bias.cuda(), conv,
               offsets, cache_indices=slots, has_initial_state=torch.ones(len(lengths), dtype=torch.bool, device='cuda'),
               activation='silu', null_block_id=-1).T
        prepared = ops.prep(mixed.contiguous(), av.cuda(), bv.cuda(), a_log.cuda(), dt.cuda(), HK, D, D, True, False)
        # Advanced indexing owns the h0 copy; the ELF aliases this copy rather
        # than shared cache state. Scatter only assigned private destinations.
        h0 = state[slots].contiguous()
        out, final = native_call(torch, ext, prepared, h0, mode,
                                offsets=offsets if len(lengths) > 1 else None)
        state[slots] = final
        return mixed, out

    mixed_a, out_a = forward(raw_a, av_a, bv_a,
                            torch.tensor([2], dtype=torch.int32, device='cuda'), [17])
    ref_a, _ = conv_oracle(torch, raw_a, weights, bias, checkpoint_conv.cpu())
    out_ref_a, h_ref_a = delta_oracle(torch, prep_oracle(torch, ref_a, av_a, bv_a, a_log, dt), checkpoint_ssm, {17})
    compare(torch, mixed_a, ref_a, 'fork A conv', rows, kind='conv')
    compare(torch, out_a, out_ref_a, 'fork A dense native output', rows, kind='output')
    compare(torch, state[2], h_ref_a[17], 'fork A native state', rows)
    compare(torch, conv[2, :, :3], raw_a[-3:].T, 'fork A conv history', rows, exact=True)
    held = [0, 1, 3, 4, 5, 6, 7]
    compare(torch, conv[held], c_before[held], 'shared checkpoint plus held sibling conv untouched', rows, exact=True)
    compare(torch, state[held], h_before[held], 'shared checkpoint plus held sibling SSM untouched', rows, exact=True)
    # Reset private slots only, then actually execute two unequal varlen forks.
    conv[2:4] = c_before[2:4]
    state[2:4] = h_before[2:4]
    mixed, output = forward(torch.cat((raw_a, raw_b)), torch.cat((av_a, av_b)), torch.cat((bv_a, bv_b)),
                 torch.tensor([2, 3], dtype=torch.int32, device='cuda'), [17, 48])
    ref_b, _ = conv_oracle(torch, raw_b, weights, bias, checkpoint_conv.cpu())
    out_ref_b, h_ref_b = delta_oracle(torch, prep_oracle(torch, ref_b, av_b, bv_b, a_log, dt), checkpoint_ssm, {31})
    compare(torch, mixed, torch.cat((ref_a, ref_b)), 'unequal packed fork conv', rows, kind='conv')
    compare(torch, output, torch.cat((out_ref_a, out_ref_b)), 'unequal packed native outputs', rows, kind='output')
    compare(torch, state[2], h_ref_a[17], 'packed fork A state', rows)
    compare(torch, state[3], h_ref_b[31], 'packed fork B state', rows)
    compare(torch, conv[2, :, :3], raw_a[-3:].T, 'packed fork A conv history', rows, exact=True)
    compare(torch, conv[3, :, :3], raw_b[-3:].T, 'packed fork B conv history', rows, exact=True)
    held = [0, 1, 4, 5, 6, 7]
    compare(torch, conv[held], c_before[held], 'packed fork nonwriter conv untouched', rows, exact=True)
    compare(torch, state[held], h_before[held], 'packed fork nonwriter SSM untouched', rows, exact=True)


def run_gpu(args, plan):
    require(args.out is not None and args.out.parent.is_dir()
            and not args.out.exists() and not args.out.is_symlink(), 'fresh output in existing directory required')
    receipt = {'plan': plan, 'passed': False, 'gpu_executed': False,
               'full_production_state_qualified': False, 'allocator_apc_cow_qualified': False,
               'logit_trajectory_qualified': False, 'tests': []}
    started = time.monotonic()
    try:
        quiet, m, gate_args, before, mode, extension = admit(args, plan)
        receipt['before'] = before
        binding = json.loads(BINDING.read_text())
        receipt['qualification_helpers'] = {
            str(QUIET): binding['quiet_helper_sha256'],
            str(MEASUREMENT): binding['measurement_sha256']}
        # No device imports above this line. Default description never gets here.
        torch = importlib.import_module('torch')
        # This standalone CPU oracle performs many small FP64 steps. Bound its
        # intra-op threads in this process only; it is not a timing benchmark.
        torch.set_num_threads(1)
        require(torch.cuda.get_device_capability() == (7, 5), 'SM75 required')
        require(torch.cuda.mem_get_info()[0] >= 512 * 1024**2, 'at least512MiB device headroom required')
        sys.path.insert(0, str(args.source_root))
        conv = importlib.import_module('vllm.model_executor.layers.mamba.ops.causal_conv1d')
        prep = importlib.import_module('vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv')
        spec = importlib.import_module('vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating')
        for module in (conv, prep, spec):
            require(Path(module.__file__).resolve().is_relative_to(args.source_root.resolve()), 'off-root executed op')
        require(sha(extension) == json.loads(BINDING.read_text())['gdn_elf_sha256'], 'ELF changed before load')
        ext = load(extension, sha(extension), 'flash_qla_legacy_gdn')
        ops = NS(conv_fn=conv.causal_conv1d_fn, conv_update=conv.causal_conv1d_update,
                 prep=prep.fused_post_conv_prep, spec=spec.fused_sigmoid_gating_delta_rule_update)
        generator = torch.Generator(device='cpu').manual_seed(73)
        def random(*shape): return torch.randn(*shape, generator=generator).half()
        max_length = max(plan['lengths'])
        raw = random(max_length, CONV_DIM) * .1
        weights, bias = random(CONV_DIM, 4) * .2, random(CONV_DIM) * .01
        a, b = random(max_length, HV) * .1, random(max_length, HV) * .1
        a_log, dt = random(HV).float() * .1 - 2, random(HV).float() * .1
        conv0, ssm0 = random(CONV_DIM, HISTORY) * .01, random(HV, D, D).float() * .01
        expected_mixed, _ = conv_oracle(torch, raw, weights, bias, conv0)
        expected_prep = prep_oracle(torch, expected_mixed, a, b, a_log, dt)
        checkpoints = set(plan['lengths'])
        for n in plan['lengths']:
            for t in (512, 1024): checkpoints.update(chunk_ends(n, t))
        expected, states = delta_oracle(torch, expected_prep, ssm0, checkpoints)
        data = (random(8, CONV_DIM) * .1, random(8, HV) * .1, random(8, HV) * .1)
        receipt['gpu_executed'] = True
        receipt['actual_qkv_mode'] = mode
        receipt['native_elf'] = {'path': str(extension), 'sha256': sha(extension), 'loaded_directly': True}
        receipt['private_fork_executed'] = False
        for n in plan['lengths']:
            result = {}
            for threshold in (0, 1024, 512):
                ends = [n] if threshold == 0 else chunk_ends(n, threshold)
                c, h = prefix(torch, ext, ops, raw, a, b, weights, bias, a_log, dt,
                              conv0, ssm0, ends, mode, receipt['tests'], expected_mixed,
                              expected, states, packed=threshold != 0)
                result[threshold] = (c, h)
                verification(torch, ops, c, h, data, weights, bias, a_log, dt,
                             receipt['tests'], f'N{n}/threshold{threshold}')
            for threshold in (1024, 512):
                compare(torch, result[threshold][0], result[0][0], 'whole-versus-split conv state',
                        receipt['tests'], exact=True, length=n, threshold=threshold)
                compare(torch, result[threshold][1], result[0][1], 'whole-versus-split SSM state',
                        receipt['tests'], length=n, threshold=threshold)
            if n == BLOCK:
                private_fork(torch, ext, ops, args.source_root, *result[0], weights, bias,
                             a_log, dt, mode, receipt['tests'], random)
                receipt['private_fork_executed'] = True
        torch.cuda.synchronize()
        receipt['after'] = quiet.quiet_gate(m, gate_args, expected_fingerprint=before['engine_identity_fingerprint'])
        for key in ('controller_identity_file_sha256', 'native_manifest_sha256'):
            require(receipt['after'][key] == before[key], 'admission evidence changed')
        require(prepare(args.source_root, args.model, plan['lengths']) == plan, 'numerical source changed')
        require(sha(extension) == receipt['native_elf']['sha256'], 'native ELF changed during numerical')
        require(all(sha(path) == digest for path, digest in receipt['qualification_helpers'].items()),
                'admission helper changed during numerical')
        receipt['complete_boundary_matrix_executed'] = set(plan['lengths']) == set(LENGTHS)
        receipt['native_final_state_alias_checked'] = True
        receipt['passed'] = True
        return receipt
    except Exception as error:
        receipt['error_type'], receipt['error'] = type(error).__name__, str(error)[:240]
        raise
    finally:
        receipt['wall_s'] = time.monotonic() - started
        args.out.write_text(json.dumps(receipt, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--length', action='append', type=int)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--run-gpu', action='store_true')
    args = parser.parse_args()
    plan = prepare(args.source_root, args.model, args.length or LENGTHS)
    if args.run_gpu:
        run_gpu(args, plan)
    else:
        print(json.dumps({**plan, 'torch_imported': 'torch' in sys.modules}, indent=2))


if __name__ == '__main__':
    main()
