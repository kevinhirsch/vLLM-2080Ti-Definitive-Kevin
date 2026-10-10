# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stdlib contract/admission tests; never imports Torch or vLLM."""
import builtins
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

ENTRY = Path(__file__).with_name('flashqla_subblock_qual.py')
spec = importlib.util.spec_from_file_location('subblock_cpu_tested', ENTRY)
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


class Contracts(unittest.TestCase):
    def test_physical_junctions_cannot_be_crossed(self):
        for n in q.LENGTHS:
            for threshold in (512, 1024):
                with self.subTest(n=n, threshold=threshold):
                    ends = q.chunk_ends(n, threshold)
                    self.assertEqual(ends[-1], n)
                    self.assertEqual(sorted(set(ends)), ends)
                    self.assertTrue(all(0 < b-a <= threshold for a,b in zip([0,*ends], ends)))
                    for junction in range(q.BLOCK, n, q.BLOCK):
                        self.assertIn(junction, ends)

    def test_required_eagle_drop_tail_and_physical_junction_coexist(self):
        # At2B+1, state@B and@2B remain mandatory even with hash-tail@2B-16.
        ends = q.chunk_ends(7137, 512)
        for position in (3568, 7120, 7136, 7137):
            self.assertIn(position, ends)
        self.assertEqual(ends[-3:], [7120, 7136, 7137])

    def test_below_checkpoint_tail_is_private_not_published_by_fixture(self):
        self.assertEqual(q.chunk_ends(512, 512), [496, 512])
        self.assertEqual(q.chunk_ends(512, 512, partial=False), [512])
        self.assertIn(3552, q.chunk_ends(3568, 1024))

    def test_unbounded_plans_refuse(self):
        for n, threshold in ((0,512),(10**8,512),(512,0),(512,256)):
            with self.subTest(n=n, threshold=threshold), self.assertRaises(ValueError):
                q.chunk_ends(n, threshold)

    def test_non_symmetric_initial_state_discriminates_vk_from_vendor_kv(self):
        result = q.vk_discriminator()
        self.assertEqual(result['VK_state_times_k'], [-.75, -1.75])
        self.assertEqual(result['old_KV_transpose_times_k'], [-1.25, -2.])
        self.assertTrue(result['different'])
        # Independent scalar one-step example. Symmetric/zero h0 would hide
        # the layout error; here both prediction and next state distinguish it.
        h, k, value, decay, beta = [[1.,2.],[3.,5.]], [.25,-.5], [.1,.2], .9, .7
        pred_vk = [decay * sum(h[v][j]*k[j] for j in range(2)) for v in range(2)]
        pred_kv = [decay * sum(h[j][v]*k[j] for j in range(2)) for v in range(2)]
        expected_vk = [[decay*h[v][j] + beta*(value[v]-pred_vk[v])*k[j]
                        for j in range(2)] for v in range(2)]
        stale_kv = [[decay*h[j][v] + beta*k[j]*(value[v]-pred_kv[v])
                     for v in range(2)] for j in range(2)]
        self.assertNotEqual(expected_vk, stale_kv)

    def test_additive_epsilon_is_not_max_norm_epsilon(self):
        vector = [1e-4, 2e-4]
        additive = [x/math.sqrt(sum(y*y for y in vector)+1e-6) for x in vector]
        max_norm = [x/max(math.sqrt(sum(y*y for y in vector)),1e-6) for x in vector]
        self.assertGreater(abs(additive[1]-max_norm[1]), .5)

    def test_cache_dtype_uses_actual_cli_not_checkpoint_only(self):
        for argv in (b'python\0--mamba-ssm-cache-dtype\0float32\0',
                     b'python\0--mamba-cache-dtype=float16\0--mamba-ssm-cache-dtype=float32\0'):
            self.assertEqual(q.cache_dtypes(argv), {'conv':'float16','ssm':'float32'})
        for argv in (b'python\0', b'python\0--mamba-ssm-cache-dtype\0auto\0',
                     b'python\0--mamba-cache-dtype=bfloat16\0--mamba-ssm-cache-dtype=float32\0',
                     b'python\0--mamba-ssm-cache-dtype\0',
                     b'python\0--mamba-ssm-cache-dtype=float32\0--mamba-ssm-cache-dtype=float16\0'):
            with self.subTest(argv=argv), self.assertRaises(ValueError): q.cache_dtypes(argv)


class SourceBinding(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root, self.model = self.base/'root', self.base/'model'
        self.root.mkdir(); self.model.mkdir()
        (self.root/'a.py').write_text('# frozen numerical source\n')
        self.config = {'linear_num_key_heads':16,'linear_num_value_heads':48,
                       'linear_key_head_dim':128,'linear_value_head_dim':128,
                       'linear_conv_kernel_dim':4,'mamba_ssm_dtype':'float32'}
        self.binding = self.base/'binding.json'
        self.write()
        self.patcher = patch.object(q, 'BINDING', self.binding)
        self.patcher.start(); self.addCleanup(self.patcher.stop)

    def write(self):
        (self.model/'config.json').write_text(json.dumps({'text_config':self.config}))
        self.binding.write_text(json.dumps({'source_commit':q.SOURCE_COMMIT,
              'source_sha256':{'a.py':q.sha(self.root/'a.py')},
              'checkpoint_config_sha256':q.sha(self.model/'config.json')}))

    def test_description_cannot_import_device_or_make_observations(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in ('torch','vllm','requests','triton'):
                raise AssertionError('device/network import attempted')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded), patch.object(q,'admit',side_effect=AssertionError):
            plan = q.prepare(self.root, self.model, [512,3568])
        self.assertFalse(plan['gpu_executed'])
        self.assertFalse(plan['allocator_apc_cow_qualified'])
        self.assertFalse(plan['logit_trajectory_qualified'])
        self.assertEqual(plan['local_tp2']['conv_dim'],5120)
        self.assertEqual(plan['entrypoint_sha256'],q.sha(ENTRY))

    def test_source_mutation_refused(self):
        (self.root/'a.py').write_text('# changed\n')
        with self.assertRaisesRegex(ValueError,'source differs'):
            q.prepare(self.root,self.model,[512])

    def test_checkpoint_mutation_refused(self):
        (self.model/'config.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'checkpoint changed'):
            q.prepare(self.root,self.model,[512])

    def test_geometry_even_when_new_config_hash_matches_refused(self):
        self.config['linear_num_value_heads'] = 24
        self.write()
        with self.assertRaisesRegex(ValueError,'geometry changed'):
            q.prepare(self.root,self.model,[512])

    def test_duplicate_or_unbounded_matrix_refused(self):
        for lengths in ([],[512,512],[3569,10**9]):
            with self.subTest(lengths=lengths), self.assertRaises(ValueError):
                q.prepare(self.root,self.model,lengths)

    def test_helper_mutation_refused_before_import(self):
        source = self.base/'helper.py'; source.write_text('raise AssertionError("must not execute")\n')
        with self.assertRaisesRegex(ValueError,'helper changed'):
            q.load(source,'0'*64,'forbidden_test_helper')


class Admission(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.args = NS(source_root=self.base/'root',model=self.base/'model',out=self.base/'result.json')
        self.binding = json.loads(q.BINDING.read_text())
        self.identity = {'fingerprint':'owned','checkpoint':{'root':str(self.args.model)},
            'flags':{'--dtype':'float16','--tensor-parallel-size':'2','--speculative-config':'{"num_speculative_tokens":3}',
                     '--additional-config':'{"gdn_prefill_backend":"flashqla_legacy"}'}}
        self.before = {'engine_pid':123,'engine_identity_fingerprint':'owned'}
        self.quiet, self.m = Mock(), Mock()
        self.quiet.quiet_gate.return_value = self.before
        self.m.runtime_identity.return_value = self.identity
        self.m.verify_loaded_native.return_value = {'loaded':[{'relative':q.GDN_REL,'sha256':self.binding['gdn_elf_sha256']}]}
        self.env = b'VLLM_FLASHQLA_LEGACY_GDN_QKV_DTYPE=native\0IGNORED_SECRET=not-recorded\0'

    def invoke(self):
        original = Path.read_bytes
        def read(path):
            if str(path)=='/proc/123/environ': return self.env
            if str(path)=='/proc/123/cmdline': return b'python\0--mamba-ssm-cache-dtype\0float32\0'
            return original(path)
        with patch.object(q,'load',side_effect=[self.quiet,self.m]), \
             patch.object(q,'sha',return_value=self.binding['gdn_elf_sha256']), \
             patch.object(q,'prepare',return_value={'frozen':True,'lengths':[512]}), \
             patch.object(Path,'read_bytes',read), \
             patch.dict(q.os.environ,{'WINDOW_ENGINE_IDENTITY_FILE':'not-read-by-mock'}):
            return q.admit(self.args,{'frozen':True,'lengths':[512]})

    def test_matching_runtime_admitted_without_device_import(self):
        result = self.invoke()
        self.assertEqual(result[4],'native')
        self.assertNotIn('torch',sys.modules)
        self.assertEqual(result[2].native_manifest,str(self.args.source_root/'RELEASE.json'))

    def test_runtime_replacement_refused(self):
        self.identity['fingerprint']='replacement'
        with self.assertRaisesRegex(ValueError,'engine changed'):
            self.invoke()

    def test_actual_model_dtype_mtp_and_backend_required(self):
        cases=[('--dtype','bfloat16','dtype'),('--tensor-parallel-size','1','TP2'),
               ('--speculative-config','{"num_speculative_tokens":1}','MTP3'),
               ('--additional-config','{}','backend')]
        for flag,value,error in cases:
            with self.subTest(flag=flag), patch.dict(self.identity['flags'],{flag:value}):
                with self.assertRaisesRegex(ValueError,error): self.invoke()

    def test_other_checkpoint_refused(self):
        self.identity['checkpoint']['root']=str(self.base/'other-model')
        with self.assertRaisesRegex(ValueError,'checkpoint differs'): self.invoke()

    def test_cli_backend_selector_has_engineargs_precedence(self):
        self.identity['flags']['--additional-config']='{}'
        self.identity['flags']['--gdn-prefill-backend']='flashqla_legacy'
        self.invoke()
        self.identity['flags']['--additional-config']='{"gdn_prefill_backend":"flashqla_legacy"}'
        self.identity['flags']['--gdn-prefill-backend']='triton'
        with self.assertRaisesRegex(ValueError,'backend'): self.invoke()

    def test_unmapped_native_elf_refused(self):
        self.m.verify_loaded_native.return_value={'loaded':[]}
        with self.assertRaisesRegex(ValueError,'not mapped'): self.invoke()

    def test_unknown_or_ambiguous_qkv_mode_refused(self):
        for env in (b'VLLM_FLASHQLA_LEGACY_GDN_QKV_DTYPE=guess\0',
                    b'VLLM_FLASHQLA_LEGACY_GDN_QKV_DTYPE=native\0VLLM_FLASHQLA_LEGACY_GDN_QKV_DTYPE=fp32\0'):
            self.env=env
            with self.subTest(env=env),self.assertRaises(ValueError): self.invoke()

    def test_gate_failure_records_no_gpu_and_precedes_torch_import(self):
        with patch.object(q,'admit',side_effect=ValueError('not exclusive')), \
             patch.object(q.importlib,'import_module',side_effect=AssertionError('must not import device')) as imported:
            with self.assertRaisesRegex(ValueError,'not exclusive'):
                q.run_gpu(self.args,{'source':'frozen'})
        imported.assert_not_called()
        receipt=json.loads(self.args.out.read_text())
        self.assertFalse(receipt['passed']); self.assertFalse(receipt['gpu_executed'])

    def test_existing_or_symlink_output_refused_before_admission(self):
        self.args.out.write_text('retain evidence')
        with patch.object(q,'admit') as admitted:
            with self.assertRaises(ValueError): q.run_gpu(self.args,{})
            admitted.assert_not_called()
        self.assertEqual(self.args.out.read_text(),'retain evidence')
        self.args.out.unlink(); self.args.out.symlink_to(self.base/'missing')
        with self.assertRaises(ValueError): q.run_gpu(self.args,{})


if __name__ == '__main__':
    unittest.main()
