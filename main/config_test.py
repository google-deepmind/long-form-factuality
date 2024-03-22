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
python -m main.config_test
```
"""

from absl.testing import absltest

# pylint: disable=g-bad-import-order
from common import data_loader
from main import config
# pylint: enable=g-bad-import-order

ACCEPTED_METHODS = (
    'vanilla_prompting',
    'naive_factuality_prompt',
    'punt_if_unsure',
    'placeholder',
    'none',
)
ACCEPTED_MODELS = (
    'gpt_4_turbo',
    'gpt_4',
    'gpt_4_32k',
    'gpt_35_turbo',
    'gpt_35_turbo_16k',
    'claude_3_opus',
    'claude_3_sonnet',
    'claude_3_haiku',
    'claude_21',
    'claude_20',
    'claude_instant',
)
ACCEPTED_TASKS = (
    'custom',
    'longfact_concepts',
    'longfact_objects',
)


class ConfigTest(absltest.TestCase):

  def test_pipeline_settings(self) -> None:
    self.assertIsInstance(config.side_1, str)
    self.assertIn(config.side_1, ACCEPTED_METHODS)
    self.assertIsInstance(config.side_2, str)
    self.assertIn(config.side_2, ACCEPTED_METHODS)
    self.assertIsInstance(config.parallelize, bool)
    self.assertIsInstance(config.save_results, bool)

  def test_model_settings(self) -> None:
    self.assertIsInstance(config.responder_model_short, str)
    self.assertIn(config.responder_model_short, ACCEPTED_MODELS)

  def test_debug_settings(self) -> None:
    self.assertIsInstance(config.show_responder_prompts, bool)
    self.assertIsInstance(config.show_responder_responses, bool)
    self.assertIsInstance(config.save_results, bool)

  def test_data_settings(self) -> None:
    self.assertIsInstance(config.task_short, str)
    self.assertIsInstance(config.shuffle_data, bool)
    self.assertIsInstance(config.max_num_examples, int)
    self.assertTrue(
        config.max_num_examples > 0 or config.max_num_examples == -1
    )
    self.assertIsInstance(config.add_universal_postamble, bool)

  def test_forced_settings(self) -> None:
    self.assertIsInstance(config.responder_model, str)
    self.assertNotEmpty(config.responder_model)
    self.assertTrue(
        isinstance(config.task, str)
        or (
            isinstance(config.task, tuple)
            and len(config.task) == data_loader.TASK_TUPLE_LENGTH
        )
    )


if __name__ == '__main__':
  absltest.main()
