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
"""Tests for modeling.py.

Run command:
```
python -m common.modeling_test
```
"""

import contextlib
from unittest import mock

from absl.testing import absltest
import langfun as lf

# pylint: disable=g-bad-import-order
from common import modeling
# pylint: enable=g-bad-import-order

_STATIC_RESPONSE = 'static_response'
_NUM_SEQUENTIAL_RESPONSES = 3
_SEQUENTIAL_RESPONSE = [
    f'{_STATIC_RESPONSE}_{i}' for i in range(_NUM_SEQUENTIAL_RESPONSES)
]

_TEST_PROMPT = 'test_prompt'
_TEST_PROMPTS = [
    f'{_TEST_PROMPT}_{i}' for i in range(_NUM_SEQUENTIAL_RESPONSES)
]

_TEST_MODEL_NAME = 'test_model'
_TEST_TEMPERATURE = 0.9
_TEST_MAX_TOKENS = 1024
_TEST_SHOW_RESPONSES = True
_TEST_SHOW_PROMPTS = True


@contextlib.contextmanager
def dummy_context_manager():
  yield None


class ModelingTest(absltest.TestCase):

  def test_fakemodel_init_base(self) -> None:
    fake_model = modeling.FakeModel()
    self.assertIsInstance(fake_model.static_response, str)
    self.assertEmpty(fake_model.static_response)
    self.assertIsNone(fake_model.sequential_responses)
    self.assertEqual(fake_model.sequential_response_idx, 0)
    self.assertIsInstance(fake_model.model, lf.llms.Echo)

  def test_fakemodel_init_static(self) -> None:
    fake_model = modeling.FakeModel(static_response=_STATIC_RESPONSE)
    self.assertEqual(fake_model.static_response, _STATIC_RESPONSE)
    self.assertIsNone(fake_model.sequential_responses)
    self.assertEqual(fake_model.sequential_response_idx, 0)
    self.assertIsInstance(fake_model.model, lf.llms.StaticResponse)

  def test_fakemodel_init_sequential(self) -> None:
    fake_model = modeling.FakeModel(
        sequential_responses=_SEQUENTIAL_RESPONSE
    )
    self.assertIsInstance(fake_model.static_response, str)
    self.assertEmpty(fake_model.static_response)
    self.assertEqual(fake_model.sequential_responses, _SEQUENTIAL_RESPONSE)
    self.assertEqual(fake_model.sequential_response_idx, 0)
    self.assertIsInstance(fake_model.model, lf.llms.StaticSequence)

  def test_fakemodel_generate_base(self) -> None:
    fake_model = modeling.FakeModel()

    for prompt in _TEST_PROMPTS:
      actual_output = fake_model.generate(prompt)
      self.assertIsInstance(actual_output, str)
      self.assertEmpty(actual_output)

  def test_fakemodel_generate_static(self) -> None:
    fake_model = modeling.FakeModel()
    fake_model.static_response = _STATIC_RESPONSE

    for prompt in _TEST_PROMPTS:
      actual_output = fake_model.generate(prompt)
      self.assertEqual(actual_output, _STATIC_RESPONSE)

  def test_fakemodel_generate_sequential(self) -> None:
    fake_model = modeling.FakeModel()
    fake_model.sequential_responses = _SEQUENTIAL_RESPONSE

    for i, prompt in enumerate(_TEST_PROMPTS + _TEST_PROMPTS):
      actual_output = fake_model.generate(prompt)
      self.assertEqual(
          actual_output, _SEQUENTIAL_RESPONSE[i % len(_SEQUENTIAL_RESPONSE)]
      )

  @mock.patch('common.modeling.Model.load')
  def test_model_init_base(self, mock_load: mock.Mock) -> None:
    mock_load.return_value = lf.llms.Echo()
    model = modeling.Model(
        model_name=_TEST_MODEL_NAME,
        temperature=_TEST_TEMPERATURE,
        max_tokens=_TEST_MAX_TOKENS,
        show_responses=_TEST_SHOW_RESPONSES,
        show_prompts=_TEST_SHOW_PROMPTS
    )
    self.assertEqual(model.model_name, _TEST_MODEL_NAME)
    self.assertEqual(model.temperature, _TEST_TEMPERATURE)
    self.assertEqual(model.max_tokens, _TEST_MAX_TOKENS)
    self.assertEqual(model.show_responses, _TEST_SHOW_RESPONSES)
    self.assertEqual(model.show_prompts, _TEST_SHOW_PROMPTS)
    self.assertIsInstance(model.model, lf.LanguageModel)
    mock_load.assert_called_once_with(
        _TEST_MODEL_NAME, _TEST_TEMPERATURE, _TEST_MAX_TOKENS
    )

  @mock.patch('common.modeling_utils.add_format')
  @mock.patch('common.modeling.Model.load')
  @mock.patch('common.modeling_utils.get_lf_context')
  @mock.patch('langfun.LangFunc')
  def test_model_generate_langfun_model(
      self,
      mock_langfunc: mock.Mock,
      mock_get_lf_context: mock.Mock,
      mock_load: mock.Mock,
      mock_add_format: mock.Mock,
  ) -> None:
    mock_load.return_value = lf.llms.Echo()
    mock_add_format.return_value = _TEST_PROMPT
    mock_get_lf_context.return_value = dummy_context_manager()
    mock_langfunc.return_value = mock.MagicMock()
    model = modeling.Model(_TEST_MODEL_NAME)
    model.generate(_TEST_PROMPT)
    mock_load.assert_called_once()
    mock_add_format.assert_called_once()
    mock_get_lf_context.assert_called_once()
    mock_langfunc.assert_called_once()

  @mock.patch('common.modeling_utils.add_format')
  @mock.patch('common.modeling.Model.load')
  @mock.patch('common.modeling_utils.get_lf_context')
  @mock.patch('common.utils.print_color')
  @mock.patch('langfun.LangFunc')
  def test_model_generate_do_debug(
      self,
      mock_langfunc: mock.Mock,
      mock_print_color: mock.Mock,
      mock_get_lf_context: mock.Mock,
      mock_load: mock.Mock,
      mock_add_format: mock.Mock,
  ) -> None:
    mock_load.return_value = lf.llms.Echo()
    mock_add_format.return_value = _TEST_PROMPT
    mock_get_lf_context.return_value = dummy_context_manager()
    mock_langfunc.return_value = mock.MagicMock()
    model = modeling.Model(
        _TEST_MODEL_NAME, show_prompts=True, show_responses=True
    )
    model.generate(_TEST_PROMPT, do_debug=True)
    mock_load.assert_called_once()
    mock_add_format.assert_called_once()
    mock_get_lf_context.assert_called_once()
    mock_langfunc.assert_called_once()
    mock_print_color.assert_called()

  @mock.patch('common.modeling.Model.load')
  @mock.patch('common.utils.to_readable_json')
  @mock.patch('builtins.print')
  def test_print_config(
      self,
      mock_print: mock.Mock,
      mock_to_readable_json: mock.Mock,
      mock_load: mock.Mock,
  ) -> None:
    test_config_dict = {'model_name': _TEST_MODEL_NAME}
    mock_load.return_value = lf.llms.Echo()
    mock_to_readable_json.return_value = str(test_config_dict)
    model = modeling.Model(_TEST_MODEL_NAME)
    model.print_config()
    mock_load.assert_called_once()
    mock_print.assert_called_once_with(str(test_config_dict))
    mock_to_readable_json.assert_called_once()


if __name__ == '__main__':
  absltest.main()
