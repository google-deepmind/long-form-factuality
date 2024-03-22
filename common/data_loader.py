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
"""All code for loading prompts and data."""

import random
from typing import Iterator, Optional

# pylint: disable=g-bad-import-order
from common import longfact
from common import utils
# pylint: enable=g-bad-import-order

DEFAULT_CUSTOM_PROMPTS = [
    'Who is Quoc V. Le?',
    'Who is Xochitl Gomez?',
    'What happened in the first modern Olympics?',
    'What happened during the 5th Academy Awards?',
    'Tell me about the company Acorns.',
    'Tell me about the Carnegie Steel Company.',
    'Give me an introduction of the city of Bucaramanga.',
    'Give me an introduction of the city of Khartoum.',
    'Tell me about the Antiguan racer snake.',
    'Tell me about the South Philippine Dwarf Kingfisher.',
]
PROMPT_FIELD = 'prompt'
CORRECT_FIELD = 'correct_answers'
INCORRECT_FIELD = 'incorrect_answers'
TASK_TUPLE_LENGTH = 4

_NONE = 'none'
_PER_PROMPT_DATA_FIELD = 'per_prompt_data'


class DataPackage:
  """Wraps a set of prompts."""

  def __init__(self) -> None:
    self.prompts: list[str] = []
    self.correct_answers: Optional[list[list[str]]] = []
    self.incorrect_answers: Optional[list[list[str]]] = []
    self.load_path: str = ''
    self.prompt_field_name: str = ''
    self.correct_answer_field_name: str = ''
    self.incorrect_answer_field_name: str = ''

  def verify_lengths(self) -> bool:
    return (
        len(self.prompts)
        == len(self.correct_answers)
        == len(self.incorrect_answers)
    )

  def iterate(self) -> Iterator[tuple[str, str, str]]:
    return zip(self.prompts, self.correct_answers, self.incorrect_answers)

  def num_items(self) -> int:
    return -1 if not self.verify_lengths() else len(self.prompts)

  def load_from_filepath(
      self,
      filepath: str,
      prompt_field_name: str,
      correct_answer_field_name: str,
      incorrect_answer_field_name: str,
  ) -> bool:
    """Loads a file into the data package."""
    data = utils.read_from_jsonlines(filepath)

    for data_point in data:
      if prompt_field_name not in data_point:
        raise ValueError(f'Missing field: {prompt_field_name}')
      else:
        self.prompts.append(data_point[prompt_field_name])

      if not correct_answer_field_name:
        self.correct_answers.append([])
      elif correct_answer_field_name.lower() == _NONE:
        self.correct_answers.append([])
      elif correct_answer_field_name in data_point:
        correct_answers = data_point[correct_answer_field_name].split('; ')
        self.correct_answers.append(correct_answers)
      else:
        raise ValueError(f'Invalid field {correct_answer_field_name}')

      if not incorrect_answer_field_name:
        self.incorrect_answers.append([])
      elif incorrect_answer_field_name.lower() == _NONE:
        self.incorrect_answers.append([])
      elif incorrect_answer_field_name in data_point:
        incorrect_answers = data_point[incorrect_answer_field_name].split('; ')
        self.incorrect_answers.append(incorrect_answers)
      else:
        raise ValueError(f'Invalid field {incorrect_answer_field_name}')

    self.load_path = filepath
    self.prompt_field_name = prompt_field_name
    self.correct_answer_field_name = correct_answer_field_name
    self.incorrect_answer_field_name = incorrect_answer_field_name
    return self.verify_lengths()

  def load_from_results_json(self, filepath: str) -> bool:
    """Loads prompts from a saved results JSON from a run."""
    try:
      results = utils.read_json(filepath)
    except IOError as e:
      utils.maybe_print_error(e)
      return self.force_load_data(DEFAULT_CUSTOM_PROMPTS)

    # Check format of the loaded JSON file is correct
    if (
        _PER_PROMPT_DATA_FIELD in results and
        isinstance(results[_PER_PROMPT_DATA_FIELD], list) and
        results[_PER_PROMPT_DATA_FIELD] and
        isinstance(results[_PER_PROMPT_DATA_FIELD][0], dict) and
        PROMPT_FIELD in results[_PER_PROMPT_DATA_FIELD][0]
    ):
      prompts = [r[PROMPT_FIELD] for r in results[_PER_PROMPT_DATA_FIELD]]
      correct_answers, incorrect_answers = None, None

      if CORRECT_FIELD in results[_PER_PROMPT_DATA_FIELD][0]:
        correct_answers = [
            r[CORRECT_FIELD] for r in results[_PER_PROMPT_DATA_FIELD]
        ]
      if INCORRECT_FIELD in results[_PER_PROMPT_DATA_FIELD][0]:
        incorrect_answers = [
            r[INCORRECT_FIELD] for r in results[_PER_PROMPT_DATA_FIELD]
        ]

      return self.force_load_data(prompts, correct_answers, incorrect_answers)
    else:
      error_message = (
          f'Invalid JSON at {filepath}, reverting to the 10 default manual'
          ' prompts.'
      )
      utils.maybe_print_error(error_message)
      return self.force_load_data(DEFAULT_CUSTOM_PROMPTS)

  def force_load_data(
      self,
      prompts: list[str],
      correct_answers: Optional[list[list[str]]] = None,
      incorrect_answers: Optional[list[list[str]]] = None,
  ) -> bool:
    """Loads a list of prompts and answers into the data package."""
    assert prompts, 'ERROR: Must provide at least one prompt.'
    self.prompts = prompts
    self.correct_answers = (
        correct_answers if correct_answers else [[]] * len(self.prompts)
    )
    self.incorrect_answers = (
        incorrect_answers if incorrect_answers else [[]] * len(self.prompts)
    )
    return self.verify_lengths()

  def shuffle_data(self, random_seed: int) -> bool:
    """Shuffles the order of data, preserving correspondence."""
    assert self.verify_lengths()
    zipped = list(self.iterate())
    random.seed(random_seed)
    random.shuffle(zipped)
    (self.prompts, self.correct_answers, self.incorrect_answers) = zip(*zipped)
    self.prompts = list(self.prompts)
    self.correct_answers = list(self.correct_answers)
    self.incorrect_answers = list(self.incorrect_answers)
    return self.verify_lengths()

  def cap_num_examples(self, max_num_examples: int) -> bool:
    """Caps the number of examples in the data package to max_num_examples."""
    if max_num_examples <= 0 or max_num_examples >= self.num_items():
      utils.print_info('No capping applied.')
      return True

    self.prompts = self.prompts[:max_num_examples]
    self.correct_answers = self.correct_answers[:max_num_examples]
    self.incorrect_answers = self.incorrect_answers[:max_num_examples]
    return self.verify_lengths()

  def load_and_prepare(
      self,
      filepath: str,
      shuffle_data: bool,
      random_seed: int,
      max_num_examples: int,
      task: Optional[tuple[str, str, str, str] | str] = None,
  ) -> None:
    """Loads data for usage."""
    if not task:
      utils.maybe_print_error('Task not provided.')
      utils.stop_all_execution(True)

    if isinstance(task, str):
      if task.endswith('.json'):
        self.load_from_results_json(task)
      elif task.endswith('/'):
        self.force_load_data(longfact.load_datasets_from_folder(task))
      elif task == 'longfact_concepts':
        self.force_load_data(longfact.load_longfact_concepts())
      elif task == 'longfact_objects':
        self.force_load_data(longfact.load_longfact_objects())
      elif task == 'custom':
        self.force_load_data(prompts=DEFAULT_CUSTOM_PROMPTS)
      else:
        utils.maybe_print_error('Invalid task.')
        utils.stop_all_execution(True)
    elif isinstance(task, tuple) and len(task) == TASK_TUPLE_LENGTH:
      task, prompt_name, correct_answer_name, incorrect_answer_name = task
      dataset_path = f'{filepath}{task}.jsonl'
      self.load_from_filepath(
          dataset_path, prompt_name, correct_answer_name, incorrect_answer_name
      )
    else:
      utils.maybe_print_error('Invalid task.')
      utils.stop_all_execution(True)

    if self.num_items() <= 0:
      utils.maybe_print_error(f'Did not load any data with task {task}.')
      utils.stop_all_execution(True)

    if shuffle_data:
      self.shuffle_data(random_seed)

    self.cap_num_examples(max_num_examples=max_num_examples)
