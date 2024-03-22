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
"""Tests for run_eval.py.

Run command:
```
python -m eval.run_eval_test
```
"""

import copy
from unittest import mock

from absl.testing import absltest

# pylint: disable=g-bad-import-order
from common import modeling
from eval import run_eval
from eval.safe import search_augmented_factuality_eval as safe
# pylint: enable=g-bad-import-order

_TEST_PROMPT = 'TEST PROMPT'
_TEST_PROMPT_DATA = {
    'prompt': _TEST_PROMPT,
    'side1_response': 'SIDE 1 TEST RESPONSE',
    'side2_response': 'SIDE 2 TEST RESPONSE',
    'correct_answers': 'TEST CORRECT ANSWER',
    'incorrect_answers': 'TEST INCORRECT ANSWER',
}
_NUM_SUPPORTED = 10
_NUM_NOT_SUPPORTED = 5
_TEST_RATING = {
    safe.SUPPORTED_LABEL: _NUM_SUPPORTED,
    safe.IRRELEVANT_LABEL: 1,
    safe.NOT_SUPPORTED_LABEL: _NUM_NOT_SUPPORTED,
}

_TEST_OUT_PATH = '/some/output/directory/filename.json'
_TEST_MODEL = modeling.FakeModel()

_NUM_PROMPTS = 3
_TEST_POSTHOC_PER_PROMPT_DATA = [
    copy.deepcopy(_TEST_PROMPT_DATA)
    | {f'side1_{run_eval.EVAL_KEY}': _TEST_RATING}
    | {f'side2_{run_eval.EVAL_KEY}': _TEST_RATING}
    for _ in range(_NUM_PROMPTS)
]
_TEST_MAX_CLAIMS = 100
_TEST_F1_RESULT = 0.5
_TEST_RESULT_DATA = (
    {s: s for s in (run_eval.SIDE_TO_SIDE_STR.values())}
    | {'per_prompt_data': _TEST_POSTHOC_PER_PROMPT_DATA}
)


class RunEvalTest(absltest.TestCase):

  @mock.patch('eval.safe.search_augmented_factuality_eval.main')
  def test_add_rating_no_ground_truth(self, mock_main: mock.Mock) -> None:
    test_prompt_data = copy.deepcopy(_TEST_PROMPT_DATA)
    test_prompt_data.pop('correct_answers')
    test_prompt_data.pop('incorrect_answers')
    mock_main.return_value = copy.deepcopy(_TEST_RATING)
    expected_output = copy.deepcopy(test_prompt_data)
    expected_output[f'side1_{run_eval.EVAL_KEY}'] = copy.deepcopy(_TEST_RATING)
    actual_output = run_eval.add_rating(
        prompt_result=test_prompt_data,
        rater_model=_TEST_MODEL,
        eval_side1=True,
        eval_side2=False,
    )
    self.assertEqual(actual_output, expected_output)
    mock_main.assert_called_once_with(
        prompt=_TEST_PROMPT,
        response=_TEST_PROMPT_DATA['side1_response'],
        rater=_TEST_MODEL,
    )

  @mock.patch('eval.safe.search_augmented_factuality_eval.main')
  def test_add_rating_with_ground_truth(self, mock_main: mock.Mock) -> None:
    test_prompt_data = copy.deepcopy(_TEST_PROMPT_DATA)
    mock_main.return_value = copy.deepcopy(_TEST_RATING)
    expected_output = copy.deepcopy(test_prompt_data)
    expected_output[f'side1_{run_eval.EVAL_KEY}'] = copy.deepcopy(_TEST_RATING)
    actual_output = run_eval.add_rating(
        prompt_result=test_prompt_data,
        rater_model=_TEST_MODEL,
        eval_side1=True,
        eval_side2=False,
    )
    self.assertEqual(actual_output, expected_output)
    mock_main.assert_called_once_with(
        prompt=_TEST_PROMPT,
        response=_TEST_PROMPT_DATA['side1_response'],
        rater=_TEST_MODEL,
    )

  @mock.patch('eval.safe.search_augmented_factuality_eval.main')
  def test_add_rating_existing_ratings(self, mock_main: mock.Mock) -> None:
    test_prompt_data = copy.deepcopy(_TEST_PROMPT_DATA)
    test_prompt_data[f'side1_{run_eval.EVAL_KEY}'] = copy.deepcopy(_TEST_RATING)
    test_prompt_data[f'side2_{run_eval.EVAL_KEY}'] = copy.deepcopy(_TEST_RATING)
    mock_main.return_value = copy.deepcopy(_TEST_RATING)
    actual_output = run_eval.add_rating(
        prompt_result=test_prompt_data,
        rater_model=_TEST_MODEL,
        eval_side1=True,
        eval_side2=False,
    )
    mock_main.assert_not_called()
    self.assertEqual(actual_output, test_prompt_data)

  def test_add_rating_no_sides(self) -> None:
    test_prompt_data = copy.deepcopy(_TEST_PROMPT_DATA)
    actual_output = run_eval.add_rating(
        prompt_result=test_prompt_data,
        rater_model=_TEST_MODEL,
        eval_side1=False,
        eval_side2=False,
    )
    self.assertEqual(actual_output, test_prompt_data)

  def test_evaluate_data_no_sides(self) -> None:
    result_data = copy.deepcopy(_TEST_RESULT_DATA)
    run_eval.evaluate_data(
        result_data=result_data,
        rater_model=_TEST_MODEL,
        do_side1=False,
        do_side2=False,
        out_path=_TEST_OUT_PATH,
        eval_in_parallel=False,
    )
    self.assertEqual(result_data, _TEST_RESULT_DATA)

  @mock.patch('eval.run_eval.add_rating')
  @mock.patch('common.utils.save_json')
  @mock.patch('common.utils.print_progress')
  def test_evaluate_data_not_parallelized(
      self,
      mock_print_progress: mock.Mock,
      mock_save_json: mock.Mock,
      mock_add_rating: mock.Mock,
  ) -> None:
    result_data = copy.deepcopy(_TEST_RESULT_DATA)
    rating_data = copy.deepcopy(_TEST_RATING)
    rating_data[f'side1_{run_eval.EVAL_KEY}'] = copy.deepcopy(_TEST_RATING)
    rating_data[f'side2_{run_eval.EVAL_KEY}'] = copy.deepcopy(_TEST_RATING)
    mock_add_rating.return_value = rating_data
    run_eval.evaluate_data(
        result_data=result_data,
        rater_model=_TEST_MODEL,
        do_side1=True,
        do_side2=True,
        out_path=_TEST_OUT_PATH,
        eval_in_parallel=False,
        show_progress_bar=False,
    )
    mock_add_rating.assert_called()
    mock_print_progress.assert_called()
    mock_save_json.assert_called()
    self.assertEqual(
        result_data['per_prompt_data'],
        [rating_data for _ in range(_NUM_PROMPTS)],
    )

  @mock.patch('eval.run_eval.add_rating')
  @mock.patch('common.utils.save_json')
  def test_evaluate_data_parallelized(
      self, mock_save_json: mock.Mock, mock_add_rating: mock.Mock
  ) -> None:
    result_data = copy.deepcopy(_TEST_RESULT_DATA)
    rating_data = copy.deepcopy(_TEST_RATING)
    rating_data[f'side1_{run_eval.EVAL_KEY}'] = copy.deepcopy(_TEST_RATING)
    rating_data[f'side2_{run_eval.EVAL_KEY}'] = copy.deepcopy(_TEST_RATING)
    mock_add_rating.return_value = rating_data
    run_eval.evaluate_data(
        result_data=result_data,
        rater_model=_TEST_MODEL,
        do_side1=True,
        do_side2=True,
        out_path=_TEST_OUT_PATH,
        eval_in_parallel=True,
        show_progress_bar=False,
    )
    mock_add_rating.assert_called()
    mock_save_json.assert_called()
    self.assertEqual(
        result_data['per_prompt_data'],
        [rating_data for _ in range(_NUM_PROMPTS)],
    )

  def test_add_aggregation_no_ratings(self) -> None:
    per_prompt_data = copy.deepcopy(_TEST_PROMPT_DATA)
    run_eval.add_aggregation(
        per_prompt_data=per_prompt_data,
        maybe_max_claims=_TEST_MAX_CLAIMS,
        eval_key=f'side1_{run_eval.EVAL_KEY}',
    )
    self.assertEqual(per_prompt_data, _TEST_PROMPT_DATA)

  @mock.patch('eval.metric_utils.calculate_metrics')
  def test_add_aggregation(self, mock_calculate_metrics: mock.Mock) -> None:
    per_prompt_data = copy.deepcopy(_TEST_POSTHOC_PER_PROMPT_DATA)
    eval_key = f'side1_{run_eval.EVAL_KEY}'
    f1_key = f'{run_eval._F1}_{_TEST_MAX_CLAIMS}'
    mock_calculate_metrics.return_value = _TEST_F1_RESULT
    run_eval.add_aggregation(
        per_prompt_data=per_prompt_data,
        maybe_max_claims=_TEST_MAX_CLAIMS,
        eval_key=eval_key,
    )
    mock_calculate_metrics.assert_called()
    self.assertNotEqual(per_prompt_data, _TEST_POSTHOC_PER_PROMPT_DATA)

    for prompt_data in per_prompt_data:
      self.assertIn(f1_key, prompt_data[eval_key])
      self.assertEqual(prompt_data[eval_key][f1_key], _TEST_F1_RESULT)

  @mock.patch('common.utils.print_divider')
  @mock.patch('common.utils.print_info')
  def test_print_results(
      self, mock_print_info: mock.Mock, mock_print_divider: mock.Mock
  ) -> None:
    result_data = copy.deepcopy(_TEST_RESULT_DATA)
    run_eval.print_results(
        result_data=result_data, maybe_max_claims=_TEST_MAX_CLAIMS
    )
    mock_print_divider.assert_called()
    mock_print_info.assert_called()


if __name__ == '__main__':
  absltest.main()
