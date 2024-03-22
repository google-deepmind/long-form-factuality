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
"""Tests for modeling_utils.py.

Run command:
```
python -m common.modeling_utils_test
```
"""

from typing import ContextManager
from unittest import mock

from absl.testing import absltest

# pylint: disable=g-bad-import-order
from common import modeling
from common import modeling_utils
# pylint: enable=g-bad-import-order

_TEST_TEMPS = [0.0, 0.5, 1.0]
_TEST_MAX_TOKENS = [0, 1024, 4096]

_TEST_OTHER_MODEL = modeling.FakeModel()

_TEST_PROMPT = 'Here is a prompt.\n'


class ModelingUtilsTest(absltest.TestCase):

  def test_add_format_model(self) -> None:
    actual_output = modeling_utils.add_format(
        prompt=_TEST_PROMPT, model=_TEST_OTHER_MODEL, model_name='test'
    )
    self.assertIn(_TEST_PROMPT.strip(), actual_output)
    self.assertNotIn('\n', actual_output)

  def test_add_format_none(self) -> None:
    actual_output = modeling_utils.add_format(
        prompt=_TEST_PROMPT, model=_TEST_OTHER_MODEL, model_name=''
    )
    self.assertEqual(actual_output, _TEST_PROMPT)

  @mock.patch('langfun.use_settings')
  def test_get_lf_context_temp_and_tokens(
      self, mock_use_settings: mock.Mock
  ) -> None:
    mock_use_settings.return_value = mock.MagicMock()

    for temp in _TEST_TEMPS:
      for max_tokens in _TEST_MAX_TOKENS:
        actual_output = modeling_utils.get_lf_context(
            temp=temp, max_tokens=max_tokens
        )
        self.assertIsInstance(actual_output, ContextManager)
        mock_use_settings.assert_called_with(
            temperature=temp, max_tokens=max_tokens
        )

  @mock.patch('langfun.use_settings')
  def test_get_lf_context_temp_only(self, mock_use_settings: mock.Mock) -> None:
    mock_use_settings.return_value = mock.MagicMock()

    for temp in _TEST_TEMPS:
      actual_output = modeling_utils.get_lf_context(temp=temp, max_tokens=None)
      self.assertIsInstance(actual_output, ContextManager)
      mock_use_settings.assert_called_with(temperature=temp)

  @mock.patch('langfun.use_settings')
  def test_get_lf_context_tokens_only(
      self, mock_use_settings: mock.Mock
  ) -> None:
    mock_use_settings.return_value = mock.MagicMock()

    for max_tokens in _TEST_MAX_TOKENS:
      actual_output = modeling_utils.get_lf_context(
          temp=None, max_tokens=max_tokens
      )
      self.assertIsInstance(actual_output, ContextManager)
      mock_use_settings.assert_called_with(max_tokens=max_tokens)

  def test_get_lf_context_no_settings(self) -> None:
    actual_output = modeling_utils.get_lf_context(temp=None, max_tokens=None)
    self.assertIsInstance(actual_output, ContextManager)

    with actual_output as context:
      self.assertIsNone(context)


if __name__ == '__main__':
  absltest.main()
