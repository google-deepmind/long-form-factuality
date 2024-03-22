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
"""Tests for longfact.py.

Run command:
```
python -m common.longfact_test
```
"""

import os
from unittest import mock

from absl.testing import absltest

# pylint: disable=g-bad-import-order
from common import longfact
# pylint: enable=g-bad-import-order

_TEST_TOPIC = 'TEST-TOPIC'
_TEST_PATH = '/some/sample/path/'
_TEST_DATASET = longfact.LongFactDataset(_TEST_TOPIC, path=_TEST_PATH)

_TEST_PROMPT_KEY = 'PROMPT_KEY'
_TEST_PROMPT = 'PROMPT_VALUE'

_NUM_PROMPTS = 3
_TEST_PROMPTS = [f'{_TEST_PROMPT}_{i}' for i in range(_NUM_PROMPTS)]
_TEST_LONGFACT_DICT = [{_TEST_PROMPT_KEY: prompt} for prompt in _TEST_PROMPTS]


class LongfactTest(absltest.TestCase):

  def test_global_variables(self) -> None:
    self.assertIsInstance(longfact.LONGFACT_CONCEPTS_FOLDER, str)
    self.assertTrue(os.path.exists(longfact.LONGFACT_CONCEPTS_FOLDER))
    self.assertIsInstance(longfact.LONGFACT_OBJECTS_FOLDER, str)
    self.assertTrue(os.path.exists(longfact.LONGFACT_OBJECTS_FOLDER))
    self.assertIsInstance(longfact.DATASETS, list)

  def test_list_topics(self) -> None:
    listed_topics = longfact.list_topics()
    self.assertNotEmpty(listed_topics)
    self.assertLen(listed_topics, len(longfact.DATASETS))

    for topic in listed_topics:
      self.assertIsInstance(topic, str)
      self.assertNotEmpty(topic)

  @mock.patch('common.utils.read_from_jsonlines')
  def test_load_datasets_base(
      self, mock_read_from_jsonlines: mock.Mock
  ) -> None:
    mock_read_from_jsonlines.return_value = _TEST_LONGFACT_DICT
    in_datasets = [_TEST_DATASET]
    actual_output = longfact.load_datasets(
        datasets=in_datasets, prompt_key=_TEST_PROMPT_KEY
    )
    self.assertLen(actual_output, _NUM_PROMPTS)
    self.assertEqual(actual_output, _TEST_PROMPTS)
    mock_read_from_jsonlines.assert_called_once_with(_TEST_PATH)

  def test_load_datasets_invalid_directory(self) -> None:
    actual_output = longfact.load_datasets(
        datasets=[longfact.LongFactDataset(_TEST_TOPIC, path='')],
        prompt_key=_TEST_PROMPT_KEY,
    )
    self.assertEmpty(actual_output)

  @mock.patch('os.listdir')
  @mock.patch('common.longfact.load_datasets')
  def test_load_datasets_from_folder(
      self, mock_load_datasets: mock.Mock, mock_list_dir: mock.Mock
  ) -> None:
    mock_list_dir.return_value = [f'longfact_{_TEST_TOPIC}.jsonl']
    mock_load_datasets.return_value = _TEST_PROMPTS
    actual_output = longfact.load_datasets_from_folder(
        directory=_TEST_PATH, prompt_key=_TEST_PROMPT_KEY
    )
    mock_list_dir.assert_called_once_with(_TEST_PATH)
    mock_load_datasets.assert_called_once()
    self.assertLen(actual_output, _NUM_PROMPTS)
    self.assertEqual(actual_output, _TEST_PROMPTS)

  @mock.patch('common.longfact.load_datasets_from_folder')
  def test_load_longfact_concepts(
      self, mock_load_datasets_from_folder: mock.Mock
  ) -> None:
    mock_load_datasets_from_folder.return_value = _TEST_PROMPTS
    actual_output = longfact.load_longfact_concepts()
    mock_load_datasets_from_folder.assert_called_once_with(
        longfact.LONGFACT_CONCEPTS_FOLDER
    )
    self.assertLen(actual_output, _NUM_PROMPTS)
    self.assertEqual(actual_output, _TEST_PROMPTS)

  @mock.patch('common.longfact.load_datasets_from_folder')
  def test_load_longfact_objects(
      self, mock_load_datasets_from_folder: mock.Mock
  ) -> None:
    mock_load_datasets_from_folder.return_value = _TEST_PROMPTS
    actual_output = longfact.load_longfact_objects()
    mock_load_datasets_from_folder.assert_called_once_with(
        longfact.LONGFACT_OBJECTS_FOLDER
    )
    self.assertLen(actual_output, _NUM_PROMPTS)
    self.assertEqual(actual_output, _TEST_PROMPTS)


if __name__ == '__main__':
  absltest.main()
