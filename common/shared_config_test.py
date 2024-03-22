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
"""Tests for shared_config.py.

Run command:
```
python -m common.shared_config_test
```
"""

from absl.testing import absltest

# pylint: disable=g-bad-import-order
from common import data_loader
from common import shared_config
# pylint: enable=g-bad-import-order


class SharedConfigTest(absltest.TestCase):

  def test_settings(self) -> None:
    self.assertIsInstance(shared_config.prompt_postamble, str)
    self.assertNotEmpty(
        shared_config.prompt_postamble
    )  # turn off instead of deleting
    self.assertIsInstance(shared_config.openai_api_key, str)
    self.assertIsInstance(shared_config.anthropic_api_key, str)
    self.assertIsInstance(shared_config.serper_api_key, str)
    self.assertIsInstance(shared_config.random_seed, int)
    self.assertIsInstance(shared_config.model_options, dict)
    self.assertNotEmpty(shared_config.model_options)

    for key, value in shared_config.model_options.items():
      self.assertIsInstance(key, str)
      self.assertIsInstance(value, str)

    self.assertIsInstance(shared_config.model_string, dict)
    self.assertNotEmpty(shared_config.model_string)

    for key, value in shared_config.model_string.items():
      self.assertIsInstance(key, str)
      self.assertIsInstance(value, str)

    self.assertIsInstance(shared_config.task_options, dict)

    for key, value in shared_config.task_options.items():
      self.assertIsInstance(key, str)
      self.assertIsInstance(value, tuple)
      self.assertLen(value, data_loader.TASK_TUPLE_LENGTH)

    self.assertIsInstance(shared_config.root_dir, str)
    self.assertNotEmpty(shared_config.root_dir)
    self.assertIsInstance(shared_config.path_to_data, str)
    self.assertNotEmpty(shared_config.path_to_data)
    self.assertIsInstance(shared_config.path_to_result, str)
    self.assertNotEmpty(shared_config.path_to_result)

if __name__ == '__main__':
  absltest.main()
