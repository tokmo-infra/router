"""CPU-only tests of the checker itself, NOT a Rust Router contract pass.

Run: python3 -B -m unittest discover -s scripts -p 'test_check_pd_prompt_ids.py' -v
All HTTP traffic in these tests stays on 127.0.0.1; no Router binary is run.
"""

import copy
import io
import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import check_pd_prompt_ids as check


class CheckerTests(unittest.TestCase):
    def setUp(self):
        self.cases = check.scenarios()

    def test_environment_isolated_and_exact_one_only(self):
        inherited = {check.ENV_NAME: '0', 'KEEP': 'yes', 'NO_PROXY': 'remote',
                     'API_KEY_VALIDATION_URLS': 'https://must-not-be-contacted.invalid'}
        for mode, value, enabled in [('default', None, True), ('on', '1', True), ('off', '0', False)]:
            with self.subTest(mode=mode):
                env, reuse_ids = check.router_environment(mode, environ=inherited)
                self.assertEqual(env.get(check.ENV_NAME), value)
                self.assertEqual(reuse_ids, enabled)
                self.assertEqual(env['KEEP'], 'yes')
                self.assertEqual(env['NO_PROXY'], '*')
                self.assertEqual(env['no_proxy'], '*')
                self.assertNotIn('API_KEY_VALIDATION_URLS', env)
        for value in ('', 'true', 'True', '01', '1 ', ' 1', '2', '0', '1'):
            with self.subTest(value=value):
                env, enabled = check.router_environment(env_value=value, environ=inherited)
                self.assertEqual(env[check.ENV_NAME], value)
                self.assertEqual(enabled, value == '1')
        self.assertEqual(inherited[check.ENV_NAME], '0')

    def test_scenario_matrix(self):
        names = {case['name'] for case in self.cases}
        self.assertEqual(len(names), len(self.cases))
        self.assertTrue({'bool-true', 'bool-false', 'negative', 'overflow-u32', 'empty',
                         'missing-ids', 'missing-kv', 'null-kv', 'array-kv',
                         'streaming', 'streaming-bad-ids', 'u32-boundaries'} <= names)
        for path in check.INPUT_KEYS:
            self.assertTrue(any(case['path'] == path for case in self.cases))
        for case in self.cases:
            if case['path'] != check.CHAT:
                self.assertNotIn('ids', case)
                self.assertFalse(case['reject_when_on'])

    def test_prefill_flags_are_chat_only_and_client_flags_survive_off(self):
        for enabled in (True, False):
            for case in self.cases:
                with self.subTest(case=case['name'], enabled=enabled):
                    body = copy.deepcopy(case['request'])
                    body['stream'] = False
                    body.pop('stream_options', None)
                    if case['path'] == check.CHAT and enabled:
                        body['return_token_ids'] = True
                    if case['path'] == '/inference/v1/generate':
                        body['sampling_params'].update(max_tokens=1, min_tokens=1)
                    elif case['path'] == '/v1/responses':
                        body['max_output_tokens'] = 1
                    else:
                        body['max_tokens'] = 1
                        for key in ('max_completion_tokens', 'min_tokens'):
                            if key in body:
                                body[key] = 1
                    check.check_prefill(case, body, enabled)
                    if case['path'] != check.CHAT:
                        body['return_token_ids'] = True
                        with self.assertRaises(AssertionError):
                            check.check_prefill(case, body, enabled)

    def decode_body(self, case, enabled):
        body = copy.deepcopy(case['request'])
        if 'kv' in case:
            body['kv_transfer_params'] = copy.deepcopy(case['kv'])
            if enabled and case['path'] == check.CHAT:
                body['kv_transfer_params']['prompt_token_ids'] = copy.deepcopy(case['ids'])
        return body

    def test_decode_preservation_and_off_fallback(self):
        for enabled in (True, False):
            for case in self.cases:
                if enabled and case['reject_when_on']:
                    continue  # Only a real Router can establish that D is blocked.
                with self.subTest(case=case['name'], enabled=enabled):
                    body = self.decode_body(case, enabled)
                    check.check_decode(case, body, enabled)
                    for field in case['request']:
                        changed = copy.deepcopy(body)
                        del changed[field]
                        with self.assertRaises(AssertionError):
                            check.check_decode(case, changed, enabled)

    def test_decode_detects_metadata_loss_wrong_ids_and_type_coercion(self):
        case = self.cases[0]
        original = copy.deepcopy(case)
        for field in check.KV:
            body = self.decode_body(case, True)
            del body['kv_transfer_params'][field]
            with self.assertRaises(AssertionError):
                check.check_decode(case, body, True)
        body = self.decode_body(case, True)
        body['kv_transfer_params']['prompt_token_ids'] = [3, 7, 8]
        with self.assertRaises(AssertionError):
            check.check_decode(case, body, True)
        body = self.decode_body(case, False)
        body['return_token_ids'] = True
        with self.assertRaises(AssertionError):
            check.check_decode(case, body, False)
        with self.assertRaises(AssertionError):
            check.equal([True], [1], 'token IDs')
        with self.assertRaises(AssertionError):
            check.equal([7.0], [7], 'token IDs')
        self.assertEqual(case, original)

    def test_backend_records_dispatch_before_failed_assertion(self):
        state = {'cases': self.cases, 'reuse_ids': True, 'calls': [], 'errors': []}
        server = ThreadingHTTPServer(('127.0.0.1', 0), check.Backend)
        server.role, server.ranks, server.contract = 'D', 8, state
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            case = self.cases[0]
            body = self.decode_body(case, True)
            body['max_tokens'] = 1  # Simulate a Router accidentally forwarding P options.
            request = urllib.request.Request(
                f'http://127.0.0.1:{server.server_port}' + check.CHAT,
                data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with self.assertRaises(urllib.error.HTTPError) as caught:
                opener.open(request, timeout=3)
            with caught.exception as response:
                self.assertEqual(response.code, 500)
                self.assertIn('D client field max_tokens', json.load(response)['error'])
            self.assertEqual(len(state['calls']), 1)
            self.assertEqual(state['calls'][0]['role'], 'D')
            self.assertEqual(state['calls'][0]['case'], case['name'])
            self.assertEqual(len(state['errors']), 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    def test_expected_500_does_not_hide_decode_dispatch_or_fixture_failure(self):
        case = next(case for case in self.cases if case['name'] == 'bool-true')
        for prefill, decode, fixture_error in [(True, False, False), (False, False, False),
                                               (True, True, False), (True, False, True)]:
            with self.subTest(prefill=prefill, decode=decode, fixture_error=fixture_error):
                state = {'cases': [case], 'reuse_ids': True, 'calls': [], 'errors': []}
                report = {'scenarios': []}

                class RejectedRequest:
                    # Unit-test the checker controller, NOT a fake Router binary.
                    def open(self, request, timeout):
                        if prefill:
                            state['calls'].append({'role': 'P', 'case': case['name']})
                        if decode:
                            state['calls'].append({'role': 'D', 'case': case['name']})
                        if fixture_error:
                            state['errors'].append('fixture assertion failed')
                        raise urllib.error.HTTPError(request.full_url, 500, 'bad IDs', {},
                                                     io.BytesIO(b'bad IDs'))

                if prefill and not decode and not fixture_error:
                    check.exercise(RejectedRequest(), 'http://127.0.0.1:1', state, report)
                    self.assertTrue(report['scenarios'][0]['blocked'])
                else:
                    with self.assertRaises(AssertionError):
                        check.exercise(RejectedRequest(), 'http://127.0.0.1:1', state, report)


if __name__ == '__main__':
    unittest.main()
