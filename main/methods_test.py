# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for methods.py.

Run command:
```
python -m main.methods_test
```
"""

from unittest import mock

from absl.testing import absltest

# pylint: disable=g-bad-import-order
from common import modeling
from main import methods
# pylint: enable=g-bad-import-order

_TEST_PROMPT = 'TEST PROMPT'
_TEST_RESPONSE = 'TEST RESPONSE'

_FAKE_RESPONDER = modeling.FakeModel(static_response=_TEST_RESPONSE)


class MethodsTest(absltest.TestCase):

  def test_fill_format_with_prompt(self) -> None:
    for prompt_format in [methods.NAIVE_FACTUALITY_PROMPT, methods.PUNT_PROMPT]:
      output = methods.fill_format_with_prompt(prompt_format, _TEST_PROMPT)
      self.assertNotEmpty(output)
      self.assertNotStartsWith('\n', output)
      self.assertNotStartsWith(' ', output)
      self.assertNotEndsWith('\n', output)
      self.assertNotEndsWith(' ', output)
      self.assertGreater(len(output), len(_TEST_PROMPT))
      self.assertIn(_TEST_PROMPT, output)

  @mock.patch('common.modeling.FakeModel.generate')
  def test_vanilla_prompting(self, mock_generate: mock.Mock) -> None:
    mock_generate.return_value = _TEST_RESPONSE
    output = methods.vanilla_prompting(_TEST_PROMPT, _FAKE_RESPONDER)
    self.assertIsInstance(output, dict)
    self.assertLen(output, 1)
    self.assertIn(methods.RESPONSE_KEY, output)
    self.assertEqual(output[methods.RESPONSE_KEY], _TEST_RESPONSE)
    mock_generate.assert_called_once_with(_TEST_PROMPT, temperature=0)

  @mock.patch('main.methods.fill_format_with_prompt')
  @mock.patch('common.modeling.FakeModel.generate')
  def test_naive_factuality_prompt(
      self, mock_generate: mock.Mock, mock_fill_format_with_prompt: mock.Mock
  ) -> None:
    mock_fill_format_with_prompt.return_value = _TEST_PROMPT
    mock_generate.return_value = _TEST_RESPONSE
    output = methods.naive_factuality_prompt(_TEST_PROMPT, _FAKE_RESPONDER)
    self.assertIsInstance(output, dict)
    self.assertLen(output, 1)
    self.assertIn(methods.RESPONSE_KEY, output)
    self.assertEqual(output[methods.RESPONSE_KEY], _TEST_RESPONSE)
    mock_fill_format_with_prompt.assert_called_once_with(
        methods.NAIVE_FACTUALITY_PROMPT, _TEST_PROMPT
    )
    mock_generate.assert_called_once_with(_TEST_PROMPT, temperature=0)

  @mock.patch('main.methods.fill_format_with_prompt')
  @mock.patch('common.modeling.FakeModel.generate')
  def test_punt_if_unsure_no_punt(
      self, mock_generate: mock.Mock, mock_fill_format_with_prompt: mock.Mock
  ) -> None:
    assert methods.PUNTED_PLACEHOLDER not in _TEST_PROMPT
    mock_fill_format_with_prompt.return_value = _TEST_PROMPT
    mock_generate.return_value = _TEST_RESPONSE
    output = methods.punt_if_unsure(_TEST_PROMPT, _FAKE_RESPONDER)
    self.assertIsInstance(output, dict)
    self.assertLen(output, 2)
    self.assertIn(methods.RESPONSE_KEY, output)
    self.assertEqual(output[methods.RESPONSE_KEY], _TEST_RESPONSE)
    self.assertIn(methods.IDK_KEY, output)
    self.assertEqual(output[methods.IDK_KEY], False)
    mock_fill_format_with_prompt.assert_called_once_with(
        methods.PUNT_PROMPT, _TEST_PROMPT
    )
    mock_generate.assert_called_once_with(_TEST_PROMPT, temperature=0)

  @mock.patch('main.methods.fill_format_with_prompt')
  @mock.patch('common.modeling.FakeModel.generate')
  def test_punt_if_unsure_punted(
      self, mock_generate: mock.Mock, mock_fill_format_with_prompt: mock.Mock
  ) -> None:
    mock_fill_format_with_prompt.return_value = _TEST_PROMPT
    mock_generate.return_value = methods.PUNTED_PLACEHOLDER + _TEST_RESPONSE
    output = methods.punt_if_unsure(_TEST_PROMPT, _FAKE_RESPONDER)
    self.assertIsInstance(output, dict)
    self.assertLen(output, 2)
    self.assertIn(methods.RESPONSE_KEY, output)
    self.assertEqual(output[methods.RESPONSE_KEY], _TEST_RESPONSE)
    self.assertIn(methods.IDK_KEY, output)
    self.assertEqual(output[methods.IDK_KEY], True)
    mock_fill_format_with_prompt.assert_called_once_with(
        methods.PUNT_PROMPT, _TEST_PROMPT
    )
    mock_generate.assert_called_once_with(_TEST_PROMPT, temperature=0)

  @mock.patch('common.utils.maybe_print_error')
  @mock.patch('common.utils.stop_all_execution')
  def test_respond_unsupported(
      self,
      mock_stop_all_execution: mock.Mock,
      mock_maybe_print_error: mock.Mock,
  ) -> None:
    response = methods.respond(
        prompt=_TEST_PROMPT,
        responder=_FAKE_RESPONDER,
        method='SOME UNSUPPORTED METHOD',
    )
    mock_maybe_print_error.assert_called_once()
    mock_stop_all_execution.assert_called_once_with(True)
    self.assertIsInstance(response, dict)
    self.assertLen(response, 1)
    self.assertIn(methods.RESPONSE_KEY, response)
    self.assertEmpty(response[methods.RESPONSE_KEY])

  def test_respond_none(self) -> None:
    response = methods.respond(
        prompt=_TEST_PROMPT, responder=_FAKE_RESPONDER, method='none'
    )
    self.assertIsInstance(response, dict)
    self.assertLen(response, 1)
    self.assertIn(methods.RESPONSE_KEY, response)
    self.assertEmpty(response[methods.RESPONSE_KEY])

  def test_respond_placeholder(self) -> None:
    response = methods.respond(
        prompt=_TEST_PROMPT, responder=_FAKE_RESPONDER, method='placeholder'
    )
    self.assertIsInstance(response, dict)
    self.assertLen(response, 1)
    self.assertIn(methods.RESPONSE_KEY, response)
    self.assertEqual(
        response[methods.RESPONSE_KEY], methods.PLACEHOLDER_RESPONSE
    )

  @mock.patch('main.methods.punt_if_unsure')
  def test_respond_punt_if_unsure(self, mock_punt_if_unsure: mock.Mock) -> None:
    expected_output = {
        methods.RESPONSE_KEY: _TEST_RESPONSE,
        methods.IDK_KEY: False,
    }
    mock_punt_if_unsure.return_value = expected_output
    response = methods.respond(
        prompt=_TEST_PROMPT, responder=_FAKE_RESPONDER, method='punt_if_unsure'
    )
    self.assertEqual(response, expected_output)
    mock_punt_if_unsure.assert_called_once_with(_TEST_PROMPT, _FAKE_RESPONDER)

  @mock.patch('main.methods.naive_factuality_prompt')
  def test_respond_naive_factuality_prompt(
      self, mock_naive_factuality_prompt: mock.Mock
  ) -> None:
    expected_output = {methods.RESPONSE_KEY: _TEST_RESPONSE}
    mock_naive_factuality_prompt.return_value = expected_output
    response = methods.respond(
        prompt=_TEST_PROMPT,
        responder=_FAKE_RESPONDER,
        method='naive_factuality_prompt',
    )
    self.assertEqual(response, expected_output)
    mock_naive_factuality_prompt.assert_called_once_with(
        _TEST_PROMPT, _FAKE_RESPONDER
    )


if __name__ == '__main__':
  absltest.main()
