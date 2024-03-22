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
"""Tests for config.py.

Run command:
```
python -m eval.safe.config_test
```
"""

from absl.testing import absltest

# pylint: disable=g-bad-import-order
from eval.safe import config
# pylint: enable=g-bad-import-order

_SUPPORTED_MODELS = (
    'gpt_4_turbo',
    'gpt_4',
    'gpt_4_32k',
    'gpt_35_turbo',
    'gpt_35_turbo_16k',
    'claude_3_opus',
    'claude_3_sonnet',
    'claude_21',
    'claude_20',
    'claude_instant',
)
_SUPPORTED_SEARCH_TYPES = ('serper')


class ConfigTest(absltest.TestCase):

  def test_model_settings(self) -> None:
    self.assertIsInstance(config.model_short, str)
    self.assertIn(config.model_short, _SUPPORTED_MODELS)
    self.assertIsInstance(config.model_temp, float)
    self.assertGreaterEqual(config.model_temp, 0.0)
    self.assertIsInstance(config.max_tokens, int)
    self.assertGreater(config.max_tokens, 0)

  def test_search_settings(self) -> None:
    self.assertIsInstance(config.search_type, str)
    self.assertIn(config.search_type, _SUPPORTED_SEARCH_TYPES)
    self.assertIsInstance(config.num_searches, int)
    self.assertGreater(config.num_searches, 0)

  def test_autorater_settings(self) -> None:
    self.assertIsInstance(config.max_steps, int)
    self.assertGreater(config.max_steps, 0)
    self.assertIsInstance(config.max_retries, int)
    self.assertGreaterEqual(config.max_retries, 0)
    self.assertIsInstance(config.debug_safe, bool)

  def test_forced_settings(self) -> None:
    self.assertIsInstance(config.model, str)
    self.assertNotEmpty(config.model)


if __name__ == '__main__':
  absltest.main()
