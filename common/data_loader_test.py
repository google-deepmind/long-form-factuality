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
"""Tests for data_loader.py.

Run command:
```
python -m common.data_loader_test
```
"""


import copy
import os
from unittest import mock

from absl.testing import absltest

# pylint: disable=g-bad-import-order
from common import data_loader
# pylint: enable=g-bad-import-order

_TEST_PROMPTS = [
    'PROMPT 1',
    'PROMPT 2',
    'PROMPT 3',
]
_TEST_CORRECT_ANSWERS = [
    ['CORRECT ANSWER 1'],
    ['CORRECT ANSWER 2'],
    ['CORRECT ANSWER 3'],
]
_TEST_INCORRECT_ANSWERS = [
    ['INCORRECT ANSWER 1'],
    ['INCORRECT ANSWER 2'],
    ['INCORRECT ANSWER 3'],
]

_SAMPLE_DIRECTORY = '/example/directory/path/'
_SAMPLE_FILEPATH = _SAMPLE_DIRECTORY + 'some_dataset.jsonl'
_SAMPLE_PROMPT_FIELD_NAME = 'Question'
_SAMPLE_CORRECT_ANSWER_FIELD_NAME = 'Correct Answer'
_SAMPLE_INCORRECT_ANSWER_FIELD_NAME = 'Incorrect Answer'
_SAMPLE_JSONL = [
    {
        _SAMPLE_PROMPT_FIELD_NAME: prompt,
        _SAMPLE_CORRECT_ANSWER_FIELD_NAME: '; '.join(correct_answer),
        _SAMPLE_INCORRECT_ANSWER_FIELD_NAME: '; '.join(incorrect_answer),
    }
    for prompt, correct_answer, incorrect_answer in zip(
        _TEST_PROMPTS, _TEST_CORRECT_ANSWERS, _TEST_INCORRECT_ANSWERS
    )
]
_RESULTS_JSON = {
    'additional_field_1': 'additional_field_1_value',
    'additional_field_2': 'additional_field_2_value',
    'per_prompt_data': [
        {
            data_loader.PROMPT_FIELD: prompt,
            data_loader.CORRECT_FIELD: correct_answer,
            data_loader.INCORRECT_FIELD: incorrect_answer,
        } for prompt, correct_answer, incorrect_answer in zip(
            _TEST_PROMPTS, _TEST_CORRECT_ANSWERS, _TEST_INCORRECT_ANSWERS
        )
    ],
}
_INVALID_RESULTS_PATH = 'Invalid results path'

_INVALID_PROMPT_FIELD_NAME = 'Invalid prompt field name'
_INVALID_CORRECT_ANSWER_FIELD_NAME = 'Invalid correct answer field name'
_INVALID_INCORRECT_ANSWER_FIELD_NAME = 'Invalid incorrect answer field name'


class DataLoaderTest(absltest.TestCase):

  def test_init(self) -> None:
    data = data_loader.DataPackage()
    self.assertEmpty(data.prompts)
    self.assertEmpty(data.correct_answers)
    self.assertEmpty(data.incorrect_answers)
    self.assertEmpty(data.load_path)
    self.assertEmpty(data.prompt_field_name)
    self.assertEmpty(data.correct_answer_field_name)
    self.assertEmpty(data.incorrect_answer_field_name)

  def test_verify_lengths_empty(self) -> None:
    data = data_loader.DataPackage()
    self.assertTrue(data.verify_lengths())

  def test_verify_lengths_unequal(self) -> None:
    data = data_loader.DataPackage()
    data.prompts = _TEST_PROMPTS
    data.correct_answers = []
    data.incorrect_answers = _TEST_INCORRECT_ANSWERS
    self.assertFalse(data.verify_lengths())
    data.prompts = []
    data.correct_answers = _TEST_CORRECT_ANSWERS
    data.incorrect_answers = _TEST_INCORRECT_ANSWERS
    self.assertFalse(data.verify_lengths())
    data.prompts = _TEST_PROMPTS
    data.correct_answers = _TEST_CORRECT_ANSWERS
    data.incorrect_answers = []
    self.assertFalse(data.verify_lengths())

  def test_verify_lengths_equal_lengths(self) -> None:
    data = data_loader.DataPackage()
    data.prompts = _TEST_PROMPTS
    data.correct_answers = _TEST_CORRECT_ANSWERS
    data.incorrect_answers = _TEST_INCORRECT_ANSWERS
    self.assertTrue(data.verify_lengths())

  def test_iterate(self) -> None:
    data = data_loader.DataPackage()
    data.prompts = _TEST_PROMPTS
    data.correct_answers = _TEST_CORRECT_ANSWERS
    data.incorrect_answers = _TEST_INCORRECT_ANSWERS

    for i, (actual_prompt, actual_answer, actual_incorrect_answer) in enumerate(
        data.iterate()
    ):
      self.assertEqual(actual_prompt, _TEST_PROMPTS[i])
      self.assertEqual(actual_answer, _TEST_CORRECT_ANSWERS[i])
      self.assertEqual(actual_incorrect_answer, _TEST_INCORRECT_ANSWERS[i])

  def test_num_items_malformed(self) -> None:
    data = data_loader.DataPackage()
    data.prompts = []
    data.correct_answers = _TEST_CORRECT_ANSWERS
    self.assertEqual(data.num_items(), -1)

  def test_num_items_valid(self) -> None:
    data = data_loader.DataPackage()
    data.prompts = _TEST_PROMPTS
    data.correct_answers = _TEST_CORRECT_ANSWERS
    data.incorrect_answers = _TEST_INCORRECT_ANSWERS
    self.assertLen(_TEST_PROMPTS, data.num_items())
    self.assertLen(_TEST_CORRECT_ANSWERS, data.num_items())
    self.assertLen(_TEST_INCORRECT_ANSWERS, data.num_items())

  @mock.patch('common.utils.read_from_jsonlines')
  def test_load_from_filepath_valid(
      self, mock_read_from_jsonlines: mock.Mock
  ) -> None:
    mock_read_from_jsonlines.return_value = _SAMPLE_JSONL
    data = data_loader.DataPackage()
    self.assertTrue(
        data.load_from_filepath(
            filepath=_SAMPLE_FILEPATH,
            prompt_field_name=_SAMPLE_PROMPT_FIELD_NAME,
            correct_answer_field_name=_SAMPLE_CORRECT_ANSWER_FIELD_NAME,
            incorrect_answer_field_name=_SAMPLE_INCORRECT_ANSWER_FIELD_NAME,
        )
    )
    self.assertGreater(data.num_items(), 0)
    self.assertNotEmpty(data.prompts)
    self.assertNotEmpty(data.correct_answers)
    self.assertNotEmpty(data.incorrect_answers)

  @mock.patch('common.utils.read_from_jsonlines')
  def test_load_from_filepath_invalid_prompt_field_name(
      self, mock_read_from_jsonlines: mock.Mock
  ) -> None:
    mock_read_from_jsonlines.return_value = _SAMPLE_JSONL
    data = data_loader.DataPackage()
    self.assertRaises(
        ValueError,
        data.load_from_filepath,
        filepath=_SAMPLE_FILEPATH,
        prompt_field_name=_INVALID_PROMPT_FIELD_NAME,
        correct_answer_field_name=_SAMPLE_CORRECT_ANSWER_FIELD_NAME,
        incorrect_answer_field_name=_SAMPLE_INCORRECT_ANSWER_FIELD_NAME,
    )

  @mock.patch('common.utils.read_from_jsonlines')
  def test_load_from_filepath_invalid_correct_answer_field_name(
      self, mock_read_from_jsonlines: mock.Mock
  ) -> None:
    mock_read_from_jsonlines.return_value = _SAMPLE_JSONL
    data = data_loader.DataPackage()
    self.assertRaises(
        ValueError,
        data.load_from_filepath,
        filepath=_SAMPLE_FILEPATH,
        prompt_field_name=_SAMPLE_PROMPT_FIELD_NAME,
        correct_answer_field_name=_INVALID_CORRECT_ANSWER_FIELD_NAME,
        incorrect_answer_field_name=_SAMPLE_INCORRECT_ANSWER_FIELD_NAME,
    )

  @mock.patch('common.utils.read_from_jsonlines')
  def test_load_from_filepath_invalid_incorrect_answer_field_name(
      self, mock_read_from_jsonlines: mock.Mock
  ) -> None:
    mock_read_from_jsonlines.return_value = _SAMPLE_JSONL
    data = data_loader.DataPackage()
    self.assertRaises(
        ValueError,
        data.load_from_filepath,
        filepath=_SAMPLE_FILEPATH,
        prompt_field_name=_SAMPLE_PROMPT_FIELD_NAME,
        correct_answer_field_name=_SAMPLE_CORRECT_ANSWER_FIELD_NAME,
        incorrect_answer_field_name=_INVALID_INCORRECT_ANSWER_FIELD_NAME,
    )

  @mock.patch('common.utils.read_from_jsonlines')
  def test_load_from_filepath_no_answer_field_name(
      self, mock_read_from_jsonlines: mock.Mock
  ) -> None:
    mock_read_from_jsonlines.return_value = _SAMPLE_JSONL
    data = data_loader.DataPackage()

    for answer_field_name in ['', 'none', 'NONE']:
      self.assertTrue(
          data.load_from_filepath(
              filepath=_SAMPLE_FILEPATH,
              prompt_field_name=_SAMPLE_PROMPT_FIELD_NAME,
              correct_answer_field_name=answer_field_name,
              incorrect_answer_field_name=answer_field_name,
          )
      )
      self.assertGreater(data.num_items(), 0)

      for answer in data.correct_answers:
        self.assertEmpty(answer)

      for answer in data.incorrect_answers:
        self.assertEmpty(answer)

  @mock.patch('common.utils.read_json')
  def test_load_from_results_json_valid(
      self, mock_read_json: mock.Mock
  ) -> None:
    mock_read_json.return_value = _RESULTS_JSON
    data = data_loader.DataPackage()
    data.load_from_results_json(filepath=_SAMPLE_FILEPATH)
    self.assertGreater(data.num_items(), 0)
    self.assertNotEmpty(data.prompts)
    self.assertNotEmpty(data.correct_answers)
    self.assertNotEmpty(data.incorrect_answers)

  @mock.patch('common.utils.read_json')
  @mock.patch('common.utils.maybe_print_error')
  def test_load_from_results_json_invalid_filepath(
      self, _: mock.Mock, mock_read_json: mock.Mock
  ) -> None:
    mock_read_json.return_value = _RESULTS_JSON
    mock_read_json.side_effect = IOError
    data = data_loader.DataPackage()
    data.load_from_results_json(filepath=_INVALID_RESULTS_PATH)
    self.assertGreater(data.num_items(), 0)
    self.assertEqual(data_loader.DEFAULT_CUSTOM_PROMPTS, data.prompts)

  @mock.patch('common.utils.read_json')
  @mock.patch('common.utils.maybe_print_error')
  def test_load_from_results_json_invalid_json_format(
      self, mock_read_json: mock.Mock, _: mock.Mock
  ) -> None:
    mock_read_json.return_value = {'invalid_json': 'invalid_json'}
    data = data_loader.DataPackage()
    data.load_from_results_json(filepath=_SAMPLE_FILEPATH)
    self.assertGreater(data.num_items(), 0)
    self.assertEqual(data_loader.DEFAULT_CUSTOM_PROMPTS, data.prompts)

  @mock.patch('common.utils.maybe_print_error')
  def test_force_load_from_results_json_invalid_path(
      self, mock_maybe_print_error: mock.Mock
  ) -> None:
    data = data_loader.DataPackage()
    self.assertTrue(data.load_from_results_json(filepath=_INVALID_RESULTS_PATH))
    mock_maybe_print_error.assert_called()
    self.assertLen(data_loader.DEFAULT_CUSTOM_PROMPTS, data.num_items())

  def test_force_load_data_prompts_only(self) -> None:
    data = data_loader.DataPackage()
    self.assertTrue(data.force_load_data(prompts=_TEST_PROMPTS))
    self.assertLen(_TEST_PROMPTS, data.num_items())
    self.assertEqual(_TEST_PROMPTS, data.prompts)

    for answer in data.correct_answers:
      self.assertEmpty(answer)

    for answer in data.incorrect_answers:
      self.assertEmpty(answer)

  def test_force_load_data_prompts_and_answers(self) -> None:
    data = data_loader.DataPackage()
    self.assertTrue(
        data.force_load_data(
            prompts=_TEST_PROMPTS,
            correct_answers=_TEST_CORRECT_ANSWERS,
            incorrect_answers=_TEST_INCORRECT_ANSWERS,
        )
    )
    self.assertLen(_TEST_PROMPTS, data.num_items())
    self.assertEqual(_TEST_PROMPTS, data.prompts)
    self.assertEqual(_TEST_CORRECT_ANSWERS, data.correct_answers)
    self.assertEqual(_TEST_INCORRECT_ANSWERS, data.incorrect_answers)

  def test_shuffle_data(self) -> None:
    data = data_loader.DataPackage()
    data.prompts = _TEST_PROMPTS
    data.correct_answers = _TEST_CORRECT_ANSWERS
    data.incorrect_answers = _TEST_INCORRECT_ANSWERS
    data.load_path = _SAMPLE_FILEPATH
    data.prompt_field_name = _SAMPLE_PROMPT_FIELD_NAME
    data.answer_field_name = _SAMPLE_CORRECT_ANSWER_FIELD_NAME
    self.assertTrue(data.shuffle_data(random_seed=0))
    shuffled_1 = copy.deepcopy(data)
    data.prompts = _TEST_PROMPTS
    data.correct_answers = _TEST_CORRECT_ANSWERS
    data.incorrect_answers = _TEST_INCORRECT_ANSWERS
    self.assertTrue(data.shuffle_data(random_seed=0))
    shuffled_2 = copy.deepcopy(data)
    self.assertEqual(shuffled_1.num_items(), shuffled_2.num_items())
    self.assertEqual(shuffled_1.prompts, shuffled_2.prompts)
    self.assertEqual(shuffled_1.correct_answers, shuffled_2.correct_answers)
    self.assertEqual(shuffled_1.incorrect_answers, shuffled_2.incorrect_answers)
    self.assertEqual(shuffled_1.load_path, shuffled_2.load_path)
    self.assertEqual(shuffled_1.prompt_field_name, shuffled_2.prompt_field_name)
    self.assertEqual(
        shuffled_1.correct_answer_field_name,
        shuffled_2.correct_answer_field_name,
    )
    self.assertEqual(
        shuffled_1.incorrect_answer_field_name,
        shuffled_2.incorrect_answer_field_name,
    )

  @mock.patch('common.utils.print_info')
  def test_cap_num_examples_invalid(self, mock_print_info: mock.Mock) -> None:
    data = data_loader.DataPackage()
    data.prompts = _TEST_PROMPTS
    data.correct_answers = _TEST_CORRECT_ANSWERS
    data.incorrect_answers = _TEST_INCORRECT_ANSWERS
    self.assertTrue(data.cap_num_examples(max_num_examples=-1))
    mock_print_info.assert_called_once()
    self.assertEqual(_TEST_PROMPTS, data.prompts)
    self.assertEqual(_TEST_CORRECT_ANSWERS, data.correct_answers)
    self.assertEqual(_TEST_INCORRECT_ANSWERS, data.incorrect_answers)
    self.assertTrue(data.cap_num_examples(max_num_examples=0))
    self.assertEqual(_TEST_PROMPTS, data.prompts)
    self.assertEqual(_TEST_CORRECT_ANSWERS, data.correct_answers)
    self.assertEqual(_TEST_INCORRECT_ANSWERS, data.incorrect_answers)

  @mock.patch('common.utils.print_info')
  def test_cap_num_examples_greater(self, mock_print_info: mock.Mock) -> None:
    data = data_loader.DataPackage()
    data.prompts = _TEST_PROMPTS
    data.correct_answers = _TEST_CORRECT_ANSWERS
    data.incorrect_answers = _TEST_INCORRECT_ANSWERS
    self.assertTrue(data.cap_num_examples(max_num_examples=data.num_items()))
    mock_print_info.assert_called_once()
    self.assertEqual(_TEST_PROMPTS, data.prompts)
    self.assertEqual(_TEST_CORRECT_ANSWERS, data.correct_answers)
    self.assertEqual(_TEST_INCORRECT_ANSWERS, data.incorrect_answers)
    self.assertTrue(
        data.cap_num_examples(max_num_examples=data.num_items() + 1)
    )
    self.assertEqual(_TEST_PROMPTS, data.prompts)
    self.assertEqual(_TEST_CORRECT_ANSWERS, data.correct_answers)
    self.assertEqual(_TEST_INCORRECT_ANSWERS, data.incorrect_answers)

  def test_cap_num_examples_valid(self) -> None:
    data = data_loader.DataPackage()
    self.assertGreater(len(_TEST_PROMPTS), 1)
    data.prompts = _TEST_PROMPTS
    data.correct_answers = _TEST_CORRECT_ANSWERS
    data.incorrect_answers = _TEST_INCORRECT_ANSWERS
    self.assertTrue(
        data.cap_num_examples(max_num_examples=data.num_items() - 1)
    )
    self.assertEqual(_TEST_PROMPTS[:-1], data.prompts)
    self.assertEqual(_TEST_CORRECT_ANSWERS[:-1], data.correct_answers)
    self.assertEqual(_TEST_INCORRECT_ANSWERS[:-1], data.incorrect_answers)

  @mock.patch('common.utils.read_json')
  @mock.patch('common.utils.print_info')
  def test_load_and_prepare_filepath(
      self, mock_print_info: mock.Mock, mock_read_json: mock.Mock
  ) -> None:
    mock_read_json.return_value = _RESULTS_JSON
    data = data_loader.DataPackage()
    data.load_and_prepare(
        filepath='',
        shuffle_data=False,
        random_seed=0,
        max_num_examples=-1,
        task=os.path.join(_SAMPLE_DIRECTORY, 'results.json'),
    )
    mock_read_json.assert_called_once()
    mock_print_info.assert_called()
    self.assertGreater(data.num_items(), 0)
    self.assertEqual(_TEST_PROMPTS, data.prompts)
    self.assertEqual(_TEST_CORRECT_ANSWERS, data.correct_answers)
    self.assertEqual(_TEST_INCORRECT_ANSWERS, data.incorrect_answers)

  @mock.patch('common.longfact.load_datasets_from_folder')
  @mock.patch('common.utils.print_info')
  def test_load_and_prepare_longfact_directory(
      self,
      mock_print_info: mock.Mock,
      mock_load_datasets_from_folder: mock.Mock,
  ) -> None:
    mock_load_datasets_from_folder.return_value = _TEST_PROMPTS
    data = data_loader.DataPackage()
    data.load_and_prepare(
        filepath=_SAMPLE_DIRECTORY,
        shuffle_data=False,
        random_seed=0,
        max_num_examples=-1,
        task=_SAMPLE_DIRECTORY,
    )
    mock_load_datasets_from_folder.assert_called_once()
    mock_print_info.assert_called()
    self.assertGreater(data.num_items(), 0)
    self.assertEqual(_TEST_PROMPTS, data.prompts)

  @mock.patch('common.longfact.load_longfact_concepts')
  @mock.patch('common.utils.print_info')
  def test_load_and_prepare_longfact(
      self, mock_print_info: mock.Mock, mock_load_all: mock.Mock
  ) -> None:
    mock_load_all.return_value = _TEST_PROMPTS
    data = data_loader.DataPackage()
    data.load_and_prepare(
        filepath=_SAMPLE_DIRECTORY,
        shuffle_data=False,
        random_seed=0,
        max_num_examples=-1,
        task='longfact_concepts',
    )
    mock_load_all.assert_called_once()
    mock_print_info.assert_called()
    self.assertGreater(data.num_items(), 0)
    self.assertEqual(_TEST_PROMPTS, data.prompts)

  @mock.patch('common.longfact.load_longfact_objects')
  @mock.patch('common.utils.print_info')
  def test_load_and_prepare_longfact_objects(
      self, mock_print_info: mock.Mock, mock_load_all: mock.Mock
  ) -> None:
    mock_load_all.return_value = _TEST_PROMPTS
    data = data_loader.DataPackage()
    data.load_and_prepare(
        filepath=_SAMPLE_DIRECTORY,
        shuffle_data=False,
        random_seed=0,
        max_num_examples=-1,
        task='longfact_objects',
    )
    mock_load_all.assert_called_once()
    mock_print_info.assert_called()
    self.assertGreater(data.num_items(), 0)
    self.assertEqual(_TEST_PROMPTS, data.prompts)

  @mock.patch('common.utils.print_info')
  def test_load_and_prepare_custom(self, mock_print_info: mock.Mock) -> None:
    data = data_loader.DataPackage()
    data.load_and_prepare(
        filepath=_SAMPLE_DIRECTORY,
        shuffle_data=False,
        random_seed=0,
        max_num_examples=-1,
        task='custom',
    )
    mock_print_info.assert_called()
    self.assertGreater(data.num_items(), 0)
    self.assertEqual(data_loader.DEFAULT_CUSTOM_PROMPTS, data.prompts)

  @mock.patch('common.utils.read_from_jsonlines')
  @mock.patch('common.utils.print_info')
  def test_load_and_prepare_tuple(
      self,
      mock_print_info: mock.Mock,
      mock_read_from_jsonlines: mock.Mock,
  ) -> None:
    mock_read_from_jsonlines.return_value = _SAMPLE_JSONL
    data = data_loader.DataPackage()
    data.load_and_prepare(
        filepath=_SAMPLE_DIRECTORY,
        shuffle_data=False,
        random_seed=0,
        max_num_examples=-1,
        task=(
            _SAMPLE_FILEPATH,
            _SAMPLE_PROMPT_FIELD_NAME,
            _SAMPLE_CORRECT_ANSWER_FIELD_NAME,
            _SAMPLE_INCORRECT_ANSWER_FIELD_NAME,
        ),
    )
    mock_print_info.assert_called()
    mock_read_from_jsonlines.assert_called_once()
    self.assertGreater(data.num_items(), 0)
    self.assertEqual(_TEST_PROMPTS, data.prompts)
    self.assertEqual(_TEST_CORRECT_ANSWERS, data.correct_answers)
    self.assertEqual(_TEST_INCORRECT_ANSWERS, data.incorrect_answers)


if __name__ == '__main__':
  absltest.main()
