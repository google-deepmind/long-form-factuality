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
python -m data_creation.config_test
```
"""

from absl.testing import absltest

# pylint: disable=g-bad-import-order
from data_creation import config
# pylint: enable=g-bad-import-order

_ACCEPTED_MODELS = (
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
_ACCEPTED_SUBTASKS = ('longfact_concepts', 'longfact_objects')


class ConfigTest(absltest.TestCase):

  def test_model_settings(self) -> None:
    self.assertIsInstance(config.generator_model, str)
    self.assertIn(config.generator_model, _ACCEPTED_MODELS)
    self.assertIsInstance(config.generation_temp, float)
    self.assertGreaterEqual(config.generation_temp, 0.0)

  def test_debug_settings(self) -> None:
    self.assertIsInstance(config.generate_data_debug, bool)
    self.assertIsInstance(config.show_generator_prompts, bool)
    self.assertIsInstance(config.show_generator_responses, bool)
    self.assertFalse(
        config.generate_data_debug
        and not config.show_generator_prompts
        and not config.show_generator_responses
    )  # Don't uselessly turn on the debug mode

  def test_data_generation_settings(self) -> None:
    self.assertIsInstance(config.subtask, str)
    self.assertIn(config.subtask, _ACCEPTED_SUBTASKS)
    self.assertIsInstance(config.num_prompts_to_generate, int)
    self.assertGreater(config.num_prompts_to_generate, 0)
    self.assertIsInstance(config.max_in_context_examples, int)
    self.assertGreaterEqual(config.max_in_context_examples, 0)
    self.assertIsInstance(config.save_results, bool)

  def test_forced_settings(self) -> None:
    self.assertIsInstance(config.generator_shorthand, str)
    self.assertNotEmpty(config.generator_shorthand)
    self.assertIsInstance(config.generator, str)
    self.assertNotEmpty(config.generator)


if __name__ == '__main__':
  absltest.main()
