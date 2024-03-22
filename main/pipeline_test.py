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
python -m main.pipeline_test
```
"""

import copy
from unittest import mock

from absl.testing import absltest

# pylint: disable=g-bad-import-order
from common import data_loader
from common import modeling
from main import pipeline
# pylint: enable=g-bad-import-order

_TEST_PROMPT = 'TEST PROMPT'
_TEST_CORRECT_ANSWER = 'TEST CORRECT ANSWER'
_TEST_INCORRECT_ANSWER = 'TEST INCORRECT ANSWER'
_TEST_POSTAMBLE = 'TEST POSTAMBLE'

_FAKE_DATA_PACKAGE = data_loader.DataPackage()

_TEST_RESPONSE = 'TEST RESPONSE'
_FAKE_RESPONDER = modeling.FakeModel()

_TEST_SIDE_1_METHOD = 'SIDE 1 METHOD'
_TEST_SIDE_2_METHOD = 'SIDE 2 METHOD'

_NUM_RESULT = 3
_TEST_RESULT = {
    'prompt': _TEST_PROMPT,
    'correct_answers': _TEST_CORRECT_ANSWER,
    'incorrect_answers': _TEST_INCORRECT_ANSWER,
    'side1_response': _TEST_RESPONSE,
    'side2_response': _TEST_RESPONSE,
}
_MULTIPLE_RESULTS = [copy.deepcopy(_TEST_RESULT) for _ in range(_NUM_RESULT)]


class PipelineTest(absltest.TestCase):

  def test_maybe_add_postamble_base(self) -> None:
    actual_output = pipeline.maybe_add_postamble(
        prompt=_TEST_PROMPT,
        add_postamble=True,
        postamble_to_add=_TEST_POSTAMBLE,
        delimiter=' ',
    )
    self.assertEqual(actual_output, _TEST_PROMPT + ' ' + _TEST_POSTAMBLE)

  def test_maybe_add_postamble_not_needed(self) -> None:
    actual_output = pipeline.maybe_add_postamble(
        prompt=_TEST_PROMPT,
        add_postamble=False,
        postamble_to_add=_TEST_POSTAMBLE,
    )
    self.assertEqual(actual_output, _TEST_PROMPT)

  @mock.patch('common.utils.maybe_print_error')
  @mock.patch('common.utils.stop_all_execution')
  def test_maybe_add_postamble_existing_postamble(
      self,
      mock_stop_all_execution: mock.Mock,
      mock_maybe_print_error: mock.Mock,
  ) -> None:
    actual_output = pipeline.maybe_add_postamble(
        prompt=_TEST_PROMPT, add_postamble=True, postamble_to_add=''
    )
    self.assertIsNone(actual_output)
    mock_maybe_print_error.assert_called_once()
    mock_stop_all_execution.assert_called_once_with(True)

  @mock.patch('common.utils.print_info')
  @mock.patch('common.modeling.FakeModel.print_config')
  @mock.patch('builtins.print')
  def test_print_config(
      self,
      mock_print: mock.Mock,
      mock_print_config: mock.Mock,
      mock_print_info: mock.Mock,
  ) -> None:
    pipeline.print_config(model_name='', model=_FAKE_RESPONDER)
    mock_print_info.assert_called_once()
    mock_print_config.assert_called_once_with()
    mock_print.assert_called_once_with()

  @mock.patch('common.utils.print_divider')
  @mock.patch('common.utils.print_color')
  @mock.patch('common.utils.print_side_by_side')
  @mock.patch('main.pipeline.maybe_add_postamble')
  @mock.patch('main.methods.respond')
  def test_get_per_prompt_result(
      self,
      mock_respond: mock.Mock,
      mock_maybe_add_postamble: mock.Mock,
      mock_print_side_by_side: mock.Mock,
      mock_print_color: mock.Mock,
      mock_print_divider: mock.Mock,
  ) -> None:
    mock_maybe_add_postamble.return_value = _TEST_PROMPT
    mock_respond.return_value = {'response': _TEST_RESPONSE}
    actual_output = pipeline.get_per_prompt_result(
        prompt=_TEST_PROMPT,
        correct_answers=_TEST_CORRECT_ANSWER,
        incorrect_answers=_TEST_INCORRECT_ANSWER,
        progress='',
        responder=_FAKE_RESPONDER,
        side_1_method=_TEST_SIDE_1_METHOD,
        side_2_method=_TEST_SIDE_2_METHOD,
    )
    self.assertEqual(actual_output, _TEST_RESULT)
    mock_print_divider.assert_called_once_with()
    mock_maybe_add_postamble.assert_called()
    mock_print_color.assert_called_once()
    mock_respond.assert_has_calls([
        mock.call(_TEST_PROMPT, _FAKE_RESPONDER, _TEST_SIDE_1_METHOD),
        mock.call(_TEST_PROMPT, _FAKE_RESPONDER, _TEST_SIDE_2_METHOD),
    ])
    mock_print_side_by_side.assert_called_once()

  @mock.patch('common.utils.get_attributes')
  @mock.patch('common.utils.save_json')
  @mock.patch('common.utils.make_directory_wrapped')
  def test_save_results_base(
      self,
      make_directory_wrapped: mock.Mock,
      mock_save_json: mock.Mock,
      mock_get_attributes: mock.Mock,
  ) -> None:
    mock_get_attributes.return_value = {'attribute': 'value'}
    expected_dict = copy.deepcopy(mock_get_attributes.return_value)
    multiple_results = copy.deepcopy(_MULTIPLE_RESULTS)
    pipeline.save_results(
        results=multiple_results, additional_info={}, module=pipeline
    )
    mock_get_attributes.assert_called_once_with(pipeline)
    make_directory_wrapped.assert_called_once()
    mock_save_json.assert_called_once_with(
        pipeline.OUT_PATH, expected_dict | {'per_prompt_data': multiple_results}
    )

  @mock.patch('main.pipeline.get_per_prompt_result')
  @mock.patch('main.pipeline.save_results')
  @mock.patch('common.utils.print_info')
  @mock.patch('common.utils.print_progress')
  @mock.patch('common.utils.maybe_print_error')
  @mock.patch('common.data_loader.DataPackage.iterate')
  def test_get_results_parallelized(
      self,
      mock_iterate: mock.Mock,
      mock_maybe_print_error: mock.Mock,
      mock_print_progress: mock.Mock,
      mock_print_info: mock.Mock,
      mock_save_results: mock.Mock,
      mock_get_per_prompt_result: mock.Mock,
  ) -> None:
    mock_iterate.return_value = [
        (_TEST_PROMPT, _TEST_CORRECT_ANSWER, _TEST_INCORRECT_ANSWER)
        for _ in range(_NUM_RESULT)
    ]
    mock_get_per_prompt_result.return_value = _TEST_RESULT
    actual_output = pipeline.get_results(
        data=_FAKE_DATA_PACKAGE,
        responder=_FAKE_RESPONDER,
        start_time=0,
        parallelize_across_prompts=True,
        show_progress=False,
    )
    self.assertEqual(actual_output, _MULTIPLE_RESULTS)
    mock_print_info.assert_called_once()
    mock_get_per_prompt_result.assert_called()
    mock_iterate.assert_called()
    mock_maybe_print_error.assert_not_called()
    mock_print_progress.assert_called()
    mock_save_results.assert_not_called()

  @mock.patch('main.pipeline.get_per_prompt_result')
  @mock.patch('main.pipeline.save_results')
  @mock.patch('common.utils.print_info')
  @mock.patch('common.utils.print_progress')
  @mock.patch('common.utils.maybe_print_error')
  @mock.patch('common.data_loader.DataPackage.iterate')
  def test_get_results_unparallelized_no_save(
      self,
      mock_iterate: mock.Mock,
      mock_maybe_print_error: mock.Mock,
      mock_print_progress: mock.Mock,
      mock_print_info: mock.Mock,
      mock_save_results: mock.Mock,
      mock_get_per_prompt_result: mock.Mock,
  ) -> None:
    mock_iterate.return_value = [
        (_TEST_PROMPT, _TEST_CORRECT_ANSWER, _TEST_INCORRECT_ANSWER)
        for _ in range(_NUM_RESULT)
    ]
    mock_get_per_prompt_result.return_value = _TEST_RESULT
    actual_output = pipeline.get_results(
        data=_FAKE_DATA_PACKAGE,
        responder=_FAKE_RESPONDER,
        start_time=0,
        parallelize_across_prompts=False,
        save_results_every_step=False,
    )
    self.assertEqual(actual_output, _MULTIPLE_RESULTS)
    mock_print_info.assert_not_called()
    mock_get_per_prompt_result.assert_called()
    mock_iterate.assert_called_once()
    mock_maybe_print_error.assert_not_called()
    mock_print_progress.assert_called()
    mock_save_results.assert_not_called()

  @mock.patch('main.pipeline.get_per_prompt_result')
  @mock.patch('main.pipeline.save_results')
  @mock.patch('common.utils.print_info')
  @mock.patch('common.utils.print_progress')
  @mock.patch('common.utils.maybe_print_error')
  @mock.patch('common.data_loader.DataPackage.iterate')
  @mock.patch('time.time')
  def test_get_results_unparallelized_with_save(
      self,
      mock_time: mock.Mock,
      mock_iterate: mock.Mock,
      mock_maybe_print_error: mock.Mock,
      mock_print_progress: mock.Mock,
      mock_print_info: mock.Mock,
      mock_save_results: mock.Mock,
      mock_get_per_prompt_result: mock.Mock,
  ) -> None:
    mock_time.return_value = 0
    mock_iterate.return_value = [
        (_TEST_PROMPT, _TEST_CORRECT_ANSWER, _TEST_INCORRECT_ANSWER)
        for _ in range(_NUM_RESULT)
    ]
    mock_get_per_prompt_result.return_value = _TEST_RESULT
    actual_output = pipeline.get_results(
        data=_FAKE_DATA_PACKAGE,
        responder=_FAKE_RESPONDER,
        start_time=0,
        parallelize_across_prompts=False,
        save_results_every_step=True,
    )
    self.assertEqual(actual_output, _MULTIPLE_RESULTS)
    mock_print_info.assert_called()
    mock_get_per_prompt_result.assert_called()
    mock_iterate.assert_called_once()
    mock_maybe_print_error.assert_not_called()
    mock_print_progress.assert_not_called()
    mock_save_results.assert_called()
    mock_time.assert_called()

  @mock.patch('common.data_loader.DataPackage.load_and_prepare')
  @mock.patch('common.utils.print_info')
  def test_load_data(
      self, mock_print_info: mock.Mock, mock_load_and_prepare: mock.Mock
  ) -> None:
    actual_output = pipeline.load_data(
        filepath='/path/to/data',
        shuffle_data=True,
        random_seed=1,
        max_num_examples=10,
        task='test task'
    )
    self.assertIsInstance(actual_output, data_loader.DataPackage)
    mock_load_and_prepare.assert_called_once_with(
        filepath='/path/to/data',
        shuffle_data=True,
        random_seed=1,
        max_num_examples=10,
        task='test task',
    )
    mock_print_info.assert_called_once()

  @mock.patch('common.utils.print_info')
  @mock.patch('common.utils.print_divider')
  @mock.patch('time.time')
  def test_get_and_record_runtime(
      self,
      mock_time: mock.Mock,
      mock_print_divider: mock.Mock,
      mock_print_info: mock.Mock,
  ) -> None:
    test_time = 11111
    mock_time.return_value = test_time
    actual_output = pipeline.get_and_record_runtime(start_time=0)
    self.assertEqual(actual_output, test_time)
    mock_print_divider.assert_called_once()
    mock_print_info.assert_called_once()


if __name__ == '__main__':
  absltest.main()
