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
"""Tests for pipeline.py.

Run command:
```
python -m data_creation.pipeline_test
```
"""

import os
from unittest import mock

from absl.testing import absltest

# pylint: disable=g-bad-import-order
from common import modeling
from common import shared_config
from data_creation import config as data_creation_config
from data_creation import pipeline
# pylint: enable=g-bad-import-order

_TEST_TOPIC = 'TEST TOPIC'
_TEST_TOPIC_SAVEABLE = _TEST_TOPIC.replace(' ', '-')

_TEST_OUTPUT_DIR = '/some/sample/directory/path'

_FAKE_GENERATOR = modeling.FakeModel()
_TEST_GENERATED_PROMPTS = ['PROMPT 1', 'PROMPT 2', 'PROMPT 3']


class PipelineTest(absltest.TestCase):

  def test_find_output_name_longfact(self) -> None:
    actual_output = pipeline.find_output_name(
        topic=_TEST_TOPIC, task_name='longfact'
    )
    expected_output = f'longfact_{_TEST_TOPIC_SAVEABLE}.jsonl'
    self.assertEqual(actual_output, expected_output)

  def test_find_output_name_not_longfact(self) -> None:
    actual_output = pipeline.find_output_name(
        topic=_TEST_TOPIC, task_name=_TEST_TOPIC_SAVEABLE
    )
    expected_output = f'{_TEST_TOPIC_SAVEABLE}_{pipeline.DATE_AND_TIME}.jsonl'
    self.assertEqual(actual_output, expected_output)

  def test_find_output_folder_forced_dir(self) -> None:
    actual_output = pipeline.find_output_folder(_TEST_OUTPUT_DIR)
    self.assertEqual(actual_output, _TEST_OUTPUT_DIR)

  def test_find_output_folder_not_forced_dir(self) -> None:
    actual_output = pipeline.find_output_folder('', _TEST_TOPIC_SAVEABLE)
    expected_output = os.path.join(
        shared_config.path_to_data, (
            f'{_TEST_TOPIC_SAVEABLE}_{data_creation_config.generator_shorthand}'
            f'_{pipeline.DATE}/'
        ),
    )
    self.assertEqual(actual_output, expected_output)

  @mock.patch('common.utils.file_exists_wrapped')
  @mock.patch('common.utils.write_to_jsonlines')
  @mock.patch('common.utils.make_directory_wrapped')
  def test_save_result_not_exists(
      self,
      mock_make_directory_wrapped: mock.Mock,
      mock_write_to_jsonlines: mock.Mock,
      mock_file_exists_wrapped: mock.Mock,
  ) -> None:
    mock_file_exists_wrapped.return_value = False
    actual_output = pipeline.save_results(
        _TEST_GENERATED_PROMPTS,
        out_dir=_TEST_OUTPUT_DIR,
        out_name=_TEST_TOPIC_SAVEABLE,
        override=False,
    )
    expected_output = os.path.join(_TEST_OUTPUT_DIR, _TEST_TOPIC_SAVEABLE)
    self.assertEqual(actual_output, expected_output)
    mock_write_to_jsonlines.assert_called_once()
    mock_make_directory_wrapped.assert_called_once_with(_TEST_OUTPUT_DIR)

  @mock.patch('common.utils.write_to_jsonlines')
  @mock.patch('common.utils.maybe_print_error')
  @mock.patch('common.utils.file_exists_wrapped')
  @mock.patch('common.utils.make_directory_wrapped')
  def test_save_result_exists_override(
      self,
      mock_make_directory_wrapped: mock.Mock,
      mock_file_exists_wrapped: mock.Mock,
      mock_maybe_print_error: mock.Mock,
      mock_write_to_jsonlines: mock.Mock,
  ) -> None:
    mock_file_exists_wrapped.return_value = True
    actual_output = pipeline.save_results(
        _TEST_GENERATED_PROMPTS,
        out_dir=_TEST_OUTPUT_DIR,
        out_name=_TEST_TOPIC_SAVEABLE,
        override=True,
    )
    expected_output = os.path.join(_TEST_OUTPUT_DIR, _TEST_TOPIC_SAVEABLE)
    self.assertEqual(actual_output, expected_output)
    mock_make_directory_wrapped.assert_called_once_with(_TEST_OUTPUT_DIR)
    mock_maybe_print_error.assert_called_once()
    mock_write_to_jsonlines.assert_called_once()
    mock_file_exists_wrapped.assert_called_once()

  @mock.patch('common.utils.file_exists_wrapped')
  @mock.patch('common.utils.write_to_jsonlines')
  @mock.patch('common.utils.stop_all_execution')
  @mock.patch('common.utils.maybe_print_error')
  @mock.patch('common.utils.make_directory_wrapped')
  def test_save_result_exists_no_override(
      self,
      mock_make_directory_wrapped: mock.Mock,
      mock_maybe_print_error: mock.Mock,
      mock_stop_all_execution: mock.Mock,
      mock_write_to_jsonlines: mock.Mock,
      mock_file_exists_wrapped: mock.Mock,
  ) -> None:
    mock_file_exists_wrapped.return_value = True
    actual_output = pipeline.save_results(
        _TEST_GENERATED_PROMPTS,
        out_dir=_TEST_OUTPUT_DIR,
        out_name=_TEST_TOPIC_SAVEABLE,
        override=False,
    )
    expected_output = os.path.join(_TEST_OUTPUT_DIR, _TEST_TOPIC_SAVEABLE)
    self.assertEqual(actual_output, expected_output)
    mock_stop_all_execution.assert_called_once()
    mock_maybe_print_error.assert_called_once()
    mock_write_to_jsonlines.assert_called_once()
    mock_make_directory_wrapped.assert_called_once_with(_TEST_OUTPUT_DIR)
    mock_file_exists_wrapped.assert_called_once()

  @mock.patch('common.utils.file_exists_wrapped')
  @mock.patch('data_creation.generate_data.run')
  @mock.patch('common.utils.maybe_print_error')
  @mock.patch('common.utils.print_progress')
  @mock.patch('common.utils.print_info')
  @mock.patch('data_creation.pipeline.find_output_name')
  @mock.patch('data_creation.pipeline.save_results')
  def test_generate_prompts_for_topics_base(
      self,
      mock_save_results: mock.Mock,
      mock_find_output_name: mock.Mock,
      mock_print_info: mock.Mock,
      mock_print_progress: mock.Mock,
      mock_maybe_print_error: mock.Mock,
      mock_run: mock.Mock,
      mock_exists: mock.Mock,
  ) -> None:
    mock_run.return_value = _TEST_GENERATED_PROMPTS
    mock_find_output_name.return_value = _TEST_TOPIC_SAVEABLE
    mock_save_results.return_value = os.path.join(
        _TEST_OUTPUT_DIR, _TEST_TOPIC_SAVEABLE
    )
    pipeline.generate_prompts_for_topics(
        topics=[_TEST_TOPIC],
        generator=_FAKE_GENERATOR,
        out_folder=_TEST_OUTPUT_DIR,
        subtask='concepts',
        override_files=False,
        num_prompts_to_generate=len(_TEST_GENERATED_PROMPTS),
        do_save_results=True,
    )
    mock_run.assert_called_once()
    mock_find_output_name.assert_called()
    mock_exists.assert_not_called()
    mock_print_info.assert_called()
    mock_save_results.assert_called()
    mock_maybe_print_error.assert_not_called()
    mock_print_progress.assert_called()

  @mock.patch('common.utils.file_exists_wrapped')
  @mock.patch('data_creation.generate_data.run')
  @mock.patch('common.utils.maybe_print_error')
  @mock.patch('common.utils.print_progress')
  @mock.patch('common.utils.print_info')
  @mock.patch('data_creation.pipeline.save_results')
  def test_generate_prompts_for_topics_no_save_results(
      self,
      mock_save_results: mock.Mock,
      mock_print_info: mock.Mock,
      mock_print_progress: mock.Mock,
      mock_maybe_print_error: mock.Mock,
      mock_run: mock.Mock,
      mock_exists: mock.Mock,
  ) -> None:
    mock_run.return_value = _TEST_GENERATED_PROMPTS
    pipeline.generate_prompts_for_topics(
        topics=[_TEST_TOPIC],
        generator=_FAKE_GENERATOR,
        out_folder=_TEST_OUTPUT_DIR,
        subtask='concepts',
        override_files=False,
        num_prompts_to_generate=len(_TEST_GENERATED_PROMPTS),
        do_save_results=False,
    )
    mock_run.assert_called_once()
    mock_exists.assert_not_called()
    mock_print_info.assert_not_called()
    mock_save_results.assert_not_called()
    mock_maybe_print_error.assert_not_called()
    mock_print_progress.assert_called()


if __name__ == '__main__':
  absltest.main()
