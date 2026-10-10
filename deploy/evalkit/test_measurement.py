"""No inference: synthetic SSE, fake HTTP responses and real quality specs."""
import copy
import io
import os
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measurement as m
import run_eval
import score_only
import scoring


def response(content='done', tokens=1, tools=None):
    message = {'role': 'assistant', 'content': content}
    if tools:
        message['tool_calls'] = tools
    return {'choices': [{'message': message, 'finish_reason': 'tool_calls' if tools else 'stop'}],
            'usage': {'completion_tokens': tokens, 'prompt_tokens': 8}}


class MeasurementTests(unittest.TestCase):
    def test_output_receipt_is_content_free_and_compares_tool_semantics(self):
        original = response('private prose', tools=[{'id': 'random-a', 'function': {'name': 'f', 'arguments': '{"x":1}'}}])
        changed_id = copy.deepcopy(original)
        changed_id['choices'][0]['message']['tool_calls'][0]['id'] = 'random-b'
        changed_id['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = '{ "x": 1 }'
        first, second = m.response_receipt(original), m.response_receipt(changed_id)
        self.assertEqual(first['assistant_output_sha256'], second['assistant_output_sha256'])
        self.assertNotIn('private prose', json.dumps(first))
        self.assertNotIn('random-a', json.dumps(first))
        changed_id['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = '{"x":2}'
        self.assertNotEqual(first['assistant_output_sha256'], m.response_receipt(changed_id)['assistant_output_sha256'])
        self.assertEqual(first['finish_reason'], 'tool_calls')
        self.assertIsNone(m.response_receipt({'choices': [False]})['finish_reason'])

    def test_profile_merges_nested_thinking_and_wins_sampling(self):
        payload = run_eval.build_base_payload({'temperature': .9, 'chat_template_kwargs': {'other': 4}},
                    {'temperature': .7, 'chat_template_kwargs': {'enable_thinking': False}})
        self.assertEqual(payload['temperature'], .7)
        self.assertEqual(payload['chat_template_kwargs'], {'enable_thinking': False, 'other': 4})

    def test_generation_must_have_real_tokens_choice_and_finish(self):
        self.assertIsNone(m.generation_error(response()))
        for bad in (response(tokens=0), response(tokens=True), response('   '), {'error': {}},
                    response(None, tools=[{'function': {'name': 'f', 'arguments': '{}'}}]),
                    response(None, tools=[{'id': 'a', 'function': {'name': 'f', 'arguments': '{'}}]),
                    {'usage': {'completion_tokens': 1}, 'choices': []}, {'usage': [], 'choices': []},
                    {'usage': {'completion_tokens': 1}, 'choices': [False]},
                    {'usage': {'completion_tokens': 1}, 'choices': [{'message': {'content': 'x'}}]}):
            self.assertIsNotNone(m.generation_error(bad))

    def test_sse_timeline_keeps_prose_and_complete_tool_separate(self):
        timeline = m.StreamTimeline(10)
        timeline.byte(10.1)
        timeline.event({'choices': [{'delta': {'content': ' '}}]}, 10.2)
        timeline.event({'choices': [{'delta': {'reasoning_content': 'private'}}]}, 10.3)
        timeline.event({'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'call_a', 'function': {'name': 'lookup', 'arguments': '{"id":'}}]}}]}, 10.4)
        timeline.event({'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': '"a"}'}}]}}]}, 10.8)
        timeline.event({'choices': [{'delta': {'content': 'Done'}, 'finish_reason': 'tool_calls'}], 'usage': {'completion_tokens': 8}}, 11)
        timeline.done = True
        report = timeline.report(11.1)
        self.assertAlmostEqual(report['first_body_byte_s'], .1)
        self.assertAlmostEqual(report['first_generated_delta_s'], .2)
        self.assertAlmostEqual(report['first_prose_s'], 1)
        self.assertAlmostEqual(report['first_tool_delta_s'], .4)
        self.assertAlmostEqual(report['first_tool_complete_s'], .8)
        self.assertAlmostEqual(report['generated_gap_max_s'], .4)
        self.assertEqual(timeline.response()['choices'][0]['message']['tool_calls'][0]['function']['arguments'], '{"id":"a"}')
        self.assertNotIn('private', json.dumps(report))

    def test_continuous_usage_removes_first_bundle_token_count_assumption(self):
        timeline = m.StreamTimeline(0)
        timeline.event({'choices': [{'delta': {'content': 'several'}}], 'usage': {'completion_tokens': 4}}, 1)
        timeline.event({'choices': [{'delta': {'content': 'more'}}], 'usage': {'completion_tokens': 10}}, 2)
        self.assertEqual(timeline.report(3)['counted_generated_tokens_s'], 6)
        self.assertEqual(timeline.report(3)['generated_event_completion_tokens'], [4, 10])

    def test_partial_valid_json_does_not_claim_final_tool_complete(self):
        timeline = m.StreamTimeline(0)
        timeline.event({'choices': [{'delta': {'tool_calls': [{'id': 'a', 'function': {'name': 'f', 'arguments': '{}'}}]}}]}, 1)
        timeline.event({'choices': [{'delta': {'tool_calls': [{'function': {'arguments': 'bad'}}]}}]}, 2)
        self.assertIsNone(timeline.report(3)['first_tool_complete_s'])

    def test_http400_and_zero_tokens_fail_before_scoring(self):
        fake = Mock(status_code=400)
        fake.__enter__ = Mock(return_value=fake)
        fake.__exit__ = Mock(return_value=False)
        with patch.object(m.requests, 'post', return_value=fake):
            self.assertEqual(m.post_chat('http://localhost:8001', {})['status'], 'http_error')
        item = {'category': 'instruction', 'check': {'type': 'no_vowels'}}
        raw = {'result': {'status': 'ok', 'response': response('brrr', tokens=0)}}
        self.assertFalse(score_only.rescore_single_shot(item, raw)['passed'])

    def test_stream_done_usage_metrics_and_multiline_frames(self):
        events = [{'choices': [{'delta': {'content': 'ok'}}]},
                  {'choices': [{'delta': {}, 'finish_reason': 'stop'}]},
                  {'choices': [], 'usage': {'completion_tokens': 2}, 'metrics': {'speculative_decoding': {'num_spec_steps': 1}}}]
        wire = b''.join(b'data: '+json.dumps(e).encode()+b'\n\n' for e in events)+b'data: [DONE]\n\n'
        fake = Mock(status_code=200, headers={})
        fake.raw = io.BytesIO(wire)
        fake.__enter__ = Mock(return_value=fake)
        fake.__exit__ = Mock(return_value=False)
        with patch.object(m.requests, 'post', return_value=fake) as post:
            result = m.post_chat('http://localhost:8001', {'temperature': .7})
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['response']['metrics']['speculative_decoding']['num_spec_steps'], 1)
        self.assertTrue(post.call_args.kwargs['json']['stream_options']['include_usage'])
        fake.raw = io.BytesIO(wire.replace(b'data: [DONE]\n\n', b''))
        with patch.object(m.requests, 'post', return_value=fake):
            self.assertEqual(m.post_chat('http://localhost:8001', {})['status'], 'invalid_generation')

    def test_agentic_every_turn_gets_profile_and_saved_request(self):
        item = json.loads((run_eval.ITEMS_DIR/'agentic_chain/item_001.json').read_text())
        calls = []
        generated = [response(None, tools=[{'id': 'a', 'type': 'function', 'function': {'name': 'lookup_user', 'arguments': '{"user_id":"u_1042"}'}}]),
                     response(None, tools=[{'id': 'b', 'type': 'function', 'function': {'name': 'send_email', 'arguments': '{"to":"dana.kim@example.com"}'}}]), response('success')]
        def post(payload, timeout_s=180):
            calls.append(copy.deepcopy(payload))
            return {'status': 'ok', 'response': generated[len(calls)-1], 'wall_s': .5, 'timeline': {'first_tool_complete_s': .4}}
        with tempfile.TemporaryDirectory() as tmp, patch.object(run_eval, 'post_chat', side_effect=post):
            verdict = run_eval.run_agentic_chain_item(item, Path(tmp), {'temperature': .7, 'presence_penalty': 1.5})
            if 'score' not in verdict:
                record = json.loads((Path(tmp)/f"{item['id']}.json").read_text())
            else:
                record = verdict
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(c['temperature'] == .7 and c['presence_penalty'] == 1.5 for c in calls))
        self.assertTrue(all('request' in step for step in record['steps']))
        self.assertTrue(record['score']['passed'])

    def test_agentic_saved_zero_tokens_cannot_pass(self):
        item = {'steps': [{'final': True, 'answer_check': {'pattern': 'done'}}]}
        raw = {'steps': [{'status': 'ok', 'response': response('done', tokens=0)}]}
        self.assertFalse(scoring.score_agentic_chain(item, raw)['passed'])

    def test_duplicate_fixture_fails_before_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base/'instruction').mkdir()
            for index in (1, 2):
                (base/f'instruction/item_{index:03}.json').write_text(json.dumps({'id': 'same', 'category': 'instruction'}))
            with patch.object(run_eval, 'ITEMS_DIR', base):
                with self.assertRaisesRegex(ValueError, 'Duplicate'):
                    run_eval.load_items({'instruction'})

    def test_shipped_core_count_and_manifest_stable(self):
        if 'loop_probe_think' in run_eval.CATEGORY_ORDER:
            items = run_eval.load_items(set(run_eval.CATEGORY_ORDER)-{'loop_probe_think', 'long_ctx_deep', 'long_ctx_depth'})
        else:
            items = run_eval.load_items()
        self.assertEqual(len(items), 100)
        self.assertEqual(len({i['id'] for i in items}), 100)
        self.assertEqual(run_eval.fixture_manifest(items)['total_items'], 100)

    def test_unreachable_expired_or_busy_offline_fence_is_not_permission(self):
        fake = Mock()
        for state in ({'offline': False}, {'offline': True, 'remaining_s': 0},
                      {'offline': True, 'remaining_s': 20, 'local_active': 1}):
            fake.json.return_value = state
            with patch.object(m.requests, 'get', return_value=fake):
                with self.assertRaises(ValueError):
                    m.require_offline_window(drained=True)
        fake.raise_for_status.side_effect = m.requests.RequestException('unreachable')
        with patch.object(m.requests, 'get', return_value=fake):
            with self.assertRaises(m.requests.RequestException):
                m.require_offline_window()

    def native_fixture(self, directory, legacy=False):
        base = Path(directory)
        root = base/'release'
        library = root/'vllm/kernel.so'
        library.parent.mkdir(parents=True)
        library.write_bytes(b'synthetic ELF bytes; no executable code')
        digest = m.file_identity(library)['sha256']
        index = root/'RELEASE.files.sha256'
        index.write_text(digest+'  vllm/kernel.so\n')
        manifest = root/'RELEASE.json'
        manifest.write_text(json.dumps({'dirty': False, 'sha': 'a'*40,
                                       'so': {'vllm/kernel.so': digest}, 'files_sha256': m.file_identity(index)['sha256']}))
        proc = base/'proc'
        for pid, start, children in ((1, '111', '2'), (2, '222', '')):
            folder = proc/str(pid)
            (folder/'task'/str(pid)).mkdir(parents=True)
            (folder/'task'/str(pid)/'children').write_text(children)
            (folder/'stat').write_text(f'{pid} (worker process) '+ ' '.join(['S']+['0']*18+[start]))
            (folder/'cmdline').write_bytes(b'\0'.join([b'python', b'--long-prefill-token-threshold', b'1000'])+b'\0')
            (folder/'environ').write_bytes(b'VLLM_SCHED_SHORT_FIRST_PREFIX_AWARE=1\0VLLM_SCHED_PREFILL_SHARE_EXACT=1\0API_TOKEN=secret-test-value\0')
            (folder/'cwd').symlink_to(root)
            (folder/'maps').write_text('')
        stat = library.stat()
        maps = proc/'2/maps'
        maps.write_text(f'1000-2000 r-xp 00000000 {os.major(stat.st_dev):x}:{os.minor(stat.st_dev):x} {stat.st_ino} {library}\n')
        identity = {'pid': 1, 'start_ticks': '111', 'source': {'path': str(root)}}
        selected = None
        if legacy:
            selected = base/'snapshot.json'
            selected.write_text(json.dumps({'prod_root': str(root), 'so': {'vllm/kernel.so': digest}, 'override_b64': 'private-content-not-recorded'}))
            manifest.unlink()
        return identity, proc, library, maps, selected

    def test_native_manifest_attests_descendant_maps_and_explicit_legacy(self):
        for legacy in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                identity, proc, library, maps, manifest = self.native_fixture(tmp, legacy)
                result = m.verify_loaded_native(identity, manifest_file=manifest, proc_root=proc)
                self.assertTrue(result['verified'])
                self.assertEqual(result['loaded'][0]['pid'], 2)
                self.assertEqual(result['expected_native_files'], 1)
                self.assertNotIn('private-content', json.dumps(result))
                self.assertIn('does not prove CUDA dispatch', result['scope'])
                self.assertEqual('legacy' in result['kind'], legacy)

    def test_explicit_wrong_native_manifest_rejects_on_immutable_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            identity, proc, library, maps, _ = self.native_fixture(tmp)
            old = Path(tmp)/'snapshot.json'
            old.write_text(json.dumps({'prod_root': '/different-production-root', 'so': {}}))
            with self.assertRaises(ValueError):
                m.verify_loaded_native(identity, manifest_file=old, proc_root=proc)
            result = m.verify_loaded_native(identity, manifest_file=library.parent.parent/'RELEASE.json', proc_root=proc)
            self.assertEqual(result['kind'], 'immutable release')

    def test_legacy_venv_third_party_mapping_is_outside_native_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            identity, proc, library, maps, manifest = self.native_fixture(tmp, legacy=True)
            third_party = Path(identity['source']['path'])/'.venv/lib/python3.10/site-packages/torch/libtorch.so'
            maps.write_text(maps.read_text()+f'3000-4000 r-xp 00000000 00:00 999 {third_party}\n')
            self.assertTrue(m.verify_loaded_native(identity, manifest_file=manifest, proc_root=proc)['verified'])

    def test_legacy_without_explicit_matching_snapshot_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            identity, proc, library, maps, manifest = self.native_fixture(tmp, legacy=True)
            with self.assertRaises(FileNotFoundError):
                m.verify_loaded_native(identity, proc_root=proc)
            manifest.write_text(json.dumps({'prod_root': '/wrong-root', 'so': {}}))
            with self.assertRaises(ValueError):
                m.verify_loaded_native(identity, manifest_file=manifest, proc_root=proc)

    def test_native_hash_deleted_replaced_and_offtree_mappings_reject(self):
        for problem in ('hash', 'deleted', 'inode', 'offtree'):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                identity, proc, library, maps, _ = self.native_fixture(tmp)
                if problem == 'hash':
                    library.write_bytes(b'changed bytes')
                elif problem == 'deleted':
                    maps.write_text(maps.read_text().strip()+' (deleted)\n')
                elif problem == 'inode':
                    fields = maps.read_text().split(None, 5)
                    fields[4] = str(int(fields[4])+1)
                    maps.write_text(' '.join(fields))
                else:
                    other = Path(tmp)/'other/vllm/kernel.so'
                    maps.write_text(maps.read_text().replace(str(library), str(other)))
                with self.assertRaises(ValueError):
                    m.verify_loaded_native(identity, proc_root=proc)

    def test_controller_native_identity_must_match_pid_start_root_and_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            identity, proc, library, maps, _ = self.native_fixture(tmp)
            control = Path(tmp)/'identity.json'
            expected = {'engine_pid': 1, 'pid_start': '111', 'root': identity['source']['path'],
                        'runtime_flags': {'VLLM_SCHED_PREFILL_SHARE_EXACT': '1'}}
            control.write_text(json.dumps(expected))
            self.assertTrue(m.verify_loaded_native(identity, identity_file=control, proc_root=proc)['verified'])
            for change in ({'engine_pid': 2}, {'pid_start': 'different'}, {'root': '/wrong'},
                           {'runtime_flags': {'VLLM_SCHED_PREFILL_SHARE_EXACT': '0'}}):
                control.write_text(json.dumps({**expected, **change}))
                with self.assertRaises(ValueError):
                    m.verify_loaded_native(identity, identity_file=control, proc_root=proc)

    def test_new_scheduler_controls_record_without_unrelated_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            identity, proc, library, maps, _ = self.native_fixture(tmp)
            real_path = Path
            with patch.object(m, 'Path', side_effect=lambda value: proc if value == '/proc' else real_path(value)):
                recorded = m.runtime_identity(1)
            self.assertEqual(recorded['runtime_env']['VLLM_SCHED_SHORT_FIRST_PREFIX_AWARE'], '1')
            self.assertEqual(recorded['runtime_env']['VLLM_SCHED_PREFILL_SHARE_EXACT'], '1')
            self.assertEqual(recorded['flags']['--long-prefill-token-threshold'], '1000')
            self.assertNotIn('secret-test-value', json.dumps(recorded))

    def test_counter_bonus_and_reset_semantics(self):
        before = dict(drafts=10, draft_tokens=30, accepted_tokens=20)
        after = dict(drafts=12, draft_tokens=36, accepted_tokens=24)
        delta = m.spec_counter_delta(before, after)
        self.assertEqual(delta['accepted_draft_tokens_per_verify'], 2)
        self.assertEqual(delta['estimated_emitted_tokens_per_verify_including_bonus'], 3)
        self.assertIn('not per-request', delta['scope'])
        self.assertFalse(m.spec_counter_delta(after, before)['available'])
        self.assertFalse(m.spec_counter_delta(None, after)['available'])


if __name__ == '__main__':
    unittest.main()
