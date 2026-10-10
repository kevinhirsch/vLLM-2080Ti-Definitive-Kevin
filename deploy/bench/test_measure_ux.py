import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_ux as ux


class ProbeTests(unittest.TestCase):
    def test_prepared_synthetic_suite_has_greedy_and_halo_tools(self):
        suite = ux.prepared_cases(profile_names=('greedy', 'halo-non-thinking'))
        self.assertEqual(len(suite['cases']), 6)
        self.assertEqual(len({c['id'] for c in suite['cases']}), 6)
        self.assertEqual(suite['suite_sha256'], ux.measurement.canonical_hash(suite['cases']))
        self.assertEqual(sum(c['kind'] == 'tool' for c in suite['cases']), 2)

    def test_exact_prompt_uses_served_template_counts_without_truncation(self):
        seen = []
        def tokenize(base, payload):
            seen.append(payload)
            return len(payload['messages'][0]['content'])//7 + 11
        with patch.object(ux, 'tokenize_prompt', side_effect=tokenize):
            payload, count = ux.exact_prompt(32768, 'http://127.0.0.1:8001', {'chat_template_kwargs': {'enable_thinking': False}})
        self.assertLessEqual(abs(count-32768), 32)
        self.assertTrue(seen)
        self.assertNotIn('truncate_prompt_tokens', payload)
        self.assertIn('Continue through the checklist', payload['messages'][0]['content'])

    def test_tokenizer_errors_and_invalid_counts_fail_closed(self):
        fake = Mock()
        fake.json.return_value = {'count': 0}
        with patch.object(ux.measurement.requests, 'post', return_value=fake) as post:
            with self.assertRaisesRegex(ValueError, 'valid count'):
                ux.tokenize_prompt('http://127.0.0.1:8001', {'model': 'qwen-local', 'messages': []})
            self.assertTrue(post.call_args.args[0].endswith('/tokenize'))
            self.assertNotIn('max_tokens', post.call_args.kwargs['json'])

    def test_window_receipt_enforces_local_exact_identity(self):
        identity = dict(window='w', phase='p', engine_pid=4, root='/r', fingerprint='f')
        result = {'route_receipts': {'X-Gateway-Route': 'local', 'X-Window-Identity': json.dumps(identity)}}
        self.assertIsNone(ux.verify_receipt(result, identity))
        self.assertIsNotNone(ux.verify_receipt(result, {**identity, 'engine_pid': 5}))
        result['route_receipts']['X-Gateway-Route'] = 'remote'
        self.assertIsNotNone(ux.verify_receipt(result, identity))

    def test_credentials_require_private_file_and_do_not_persist_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            token = Path(tmp)/'token'
            identity = Path(tmp)/'identity'
            token.write_text('secret-test-token')
            identity.write_text(json.dumps(dict(window='w', phase='p', engine_pid=4, root='/r', fingerprint='f')))
            token.chmod(0o644)
            with self.assertRaisesRegex(ValueError, '0600'):
                ux.window_credentials(token, identity)
            token.chmod(0o600)
            headers, receipt = ux.window_credentials(token, identity)
            self.assertEqual(headers['X-Window-Token'], 'secret-test-token')
            self.assertNotIn('secret-test-token', json.dumps(receipt))

    def test_wave_rates_exclude_invalid_requests(self):
        rows = [{'valid': False, 'usage': {'completion_tokens': 999}, 'timeline': {}},
                {'valid': True, 'usage': {'completion_tokens': 10}, 'timeline': {'first_useful_candidate_s': .5}, 'estimated_decode_tokens_s': 8}]
        summary = ux.wave_summary(rows, 2, 2)
        self.assertEqual(summary['aggregate_end_to_end_tokens_s'], 5)
        self.assertEqual(summary['valid_requests'], 1)


if __name__ == '__main__':
    unittest.main()
