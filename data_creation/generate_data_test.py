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
"""Tests for generate_data.py.

Run command:
```
python -m data_creation.generate_data_test
```
"""

from unittest import mock

from absl.testing import absltest

# pylint: disable=g-bad-import-order
from common import modeling
from data_creation import examples
from data_creation import generate_data
# pylint: enable=g-bad-import-order

_TEST_TOPIC = 'TEST TOPIC'
_TEST_TOPICS = [
    _TEST_TOPIC,
    f'\n\n{_TEST_TOPIC}\n\n',
    f'!@#$%^{_TEST_TOPIC})%(^)',
    f'  {_TEST_TOPIC}  ',
]

_QUESTIONS = [
    'What is 1 + 1?',
    'What is one plus one?',
    '2 + 2 = ?',
]
_QUESTION_GENERATOR = modeling.FakeModel(
    sequential_responses=[f'[{question}]' for question in _QUESTIONS]
)


class GenerateDataTest(absltest.TestCase):

  def test_construct_prompt_concepts(self) -> None:
    for topic in _TEST_TOPICS:
      constructed_prompt = generate_data.construct_prompt(
          topic, examples_list=examples.CONCEPT_EXAMPLES, subtask='concepts'
      )
      self.assertNotEmpty(constructed_prompt)
      self.assertIn(_TEST_TOPIC, constructed_prompt)
      self.assertNotStartsWith(constructed_prompt, '\n')
      self.assertNotEndsWith(constructed_prompt, '\n')
      self.assertGreaterEqual(len(constructed_prompt), len(_TEST_TOPIC))

  def test_construct_prompt_objects(self) -> None:
    for topic in _TEST_TOPICS:
      constructed_prompt = generate_data.construct_prompt(
          topic, examples_list=examples.OBJECT_EXAMPLES, subtask='objects'
      )
      self.assertNotEmpty(constructed_prompt)
      self.assertIn(_TEST_TOPIC, constructed_prompt)
      self.assertNotStartsWith(constructed_prompt, '\n')
      self.assertNotEndsWith(constructed_prompt, '\n')
      self.assertGreaterEqual(len(constructed_prompt), len(_TEST_TOPIC))

  def test_generate_single_prompt(self) -> None:
    for expected_question in _QUESTIONS:
      actual_output = generate_data.generate_single_prompt(
          topic=_TEST_TOPIC,
          model=_QUESTION_GENERATOR,
          examples_list=examples.CONCEPT_EXAMPLES,
          subtask='concepts',
      )
      self.assertEqual(actual_output, expected_question)
    for expected_question in _QUESTIONS:
      actual_output = generate_data.generate_single_prompt(
          topic=_TEST_TOPIC,
          model=_QUESTION_GENERATOR,
          examples_list=examples.OBJECT_EXAMPLES,
          subtask='objects',
      )
      self.assertEqual(actual_output, expected_question)

  @mock.patch('common.utils.print_progress')
  def test_run_concepts(
      self, mock_print_progress: mock.Mock
  ) -> None:
    actual_output = generate_data.run(
        topic=_TEST_TOPIC,
        generator=_QUESTION_GENERATOR,
        num_prompts=len(_QUESTIONS),
        subtask='concepts',
    )
    mock_print_progress.assert_called()
    self.assertEqual(len(actual_output), len(_QUESTIONS))
    self.assertEqual(set(actual_output), set(_QUESTIONS))

  @mock.patch('common.utils.print_progress')
  def test_run_objects(
      self, mock_print_progress: mock.Mock
  ) -> None:
    actual_output = generate_data.run(
        topic=_TEST_TOPIC,
        generator=_QUESTION_GENERATOR,
        num_prompts=len(_QUESTIONS),
        subtask='objects',
    )
    mock_print_progress.assert_called()
    self.assertEqual(len(actual_output), len(_QUESTIONS))
    self.assertEqual(set(actual_output), set(_QUESTIONS))


if __name__ == '__main__':
  absltest.main()
