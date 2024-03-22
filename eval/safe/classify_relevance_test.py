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
"""Tests for classify_relevance.py.

Run command:
```
python -m eval.safe.classify_relevance_test
```
"""

from unittest import mock

from absl.testing import absltest

# pylint: disable=g-bad-import-order
from common import modeling
from eval.safe import classify_relevance
# pylint: enable=g-bad-import-order

_TEST_PROMPT = 'This is a question.'
_TEST_RESPONSE = 'This was the full response to the question.'
_TEST_STATEMENT = 'This is a statement from the response.'
_TEST_REVISED_STATEMENT = 'This is a revised version of the statement.'

_TEST_PROMPT_TO_SEND = 'Here are some instructions. Follow them.'
_TEST_RESPONSE_TO_RECEIVE = 'Here is a response following the instructions.'
_TEST_MODEL = modeling.FakeModel()


class ClassifyRelevanceTest(absltest.TestCase):

  @mock.patch('common.utils.strip_string')
  @mock.patch('common.modeling.FakeModel.generate')
  @mock.patch('common.utils.extract_first_square_brackets')
  def test_check_relevance_is_relevant(
      self,
      mock_extract_first_square_brackets: mock.Mock,
      mock_generate: mock.Mock,
      mock_strip_string: mock.Mock,
  ) -> None:
    mock_strip_string.return_value = _TEST_PROMPT_TO_SEND
    mock_generate.return_value = _TEST_RESPONSE_TO_RECEIVE
    mock_extract_first_square_brackets.return_value = classify_relevance.SYMBOL
    actual_model_response, actual_answer = classify_relevance.check_relevance(
        prompt=_TEST_PROMPT,
        response=_TEST_RESPONSE,
        atomic_fact=_TEST_REVISED_STATEMENT,
        model=_TEST_MODEL,
        do_debug=False,
        max_retries=1,
    )
    mock_strip_string.assert_called_once()
    mock_generate.assert_called_once_with(_TEST_PROMPT_TO_SEND, do_debug=False)
    mock_extract_first_square_brackets.assert_called_once_with(
        _TEST_RESPONSE_TO_RECEIVE
    )
    self.assertEqual(actual_model_response, _TEST_RESPONSE_TO_RECEIVE)
    self.assertTrue(actual_answer)

  @mock.patch('common.utils.strip_string')
  @mock.patch('common.modeling.FakeModel.generate')
  @mock.patch('common.utils.extract_first_square_brackets')
  def test_check_relevance_is_not_relevant(
      self,
      mock_extract_first_square_brackets: mock.Mock,
      mock_generate: mock.Mock,
      mock_strip_string: mock.Mock,
  ) -> None:
    mock_strip_string.return_value = _TEST_PROMPT_TO_SEND
    mock_generate.return_value = _TEST_RESPONSE_TO_RECEIVE
    mock_extract_first_square_brackets.return_value = (
        classify_relevance.NOT_SYMBOL
    )
    actual_model_response, actual_answer = classify_relevance.check_relevance(
        prompt=_TEST_PROMPT,
        response=_TEST_RESPONSE,
        atomic_fact=_TEST_REVISED_STATEMENT,
        model=_TEST_MODEL,
        do_debug=False,
        max_retries=1,
    )
    mock_strip_string.assert_called_once()
    mock_generate.assert_called_once_with(_TEST_PROMPT_TO_SEND, do_debug=False)
    mock_extract_first_square_brackets.assert_called_once_with(
        _TEST_RESPONSE_TO_RECEIVE
    )
    self.assertEqual(actual_model_response, _TEST_RESPONSE_TO_RECEIVE)
    self.assertFalse(actual_answer)

  @mock.patch('common.utils.strip_string')
  @mock.patch('common.modeling.FakeModel.generate')
  @mock.patch('common.utils.extract_first_square_brackets')
  def test_check_relevance_not_parseable_returns_relevant(
      self,
      mock_extract_first_square_brackets: mock.Mock,
      mock_generate: mock.Mock,
      mock_strip_string: mock.Mock,
  ) -> None:
    mock_strip_string.return_value = _TEST_PROMPT_TO_SEND
    mock_generate.return_value = _TEST_RESPONSE_TO_RECEIVE
    mock_extract_first_square_brackets.return_value = ''
    actual_model_response, actual_answer = classify_relevance.check_relevance(
        prompt=_TEST_PROMPT,
        response=_TEST_RESPONSE,
        atomic_fact=_TEST_REVISED_STATEMENT,
        model=_TEST_MODEL,
        do_debug=False,
        max_retries=0,
    )
    mock_strip_string.assert_called_once()
    mock_generate.assert_called_once_with(_TEST_PROMPT_TO_SEND, do_debug=False)
    mock_extract_first_square_brackets.assert_called_once_with(
        _TEST_RESPONSE_TO_RECEIVE
    )
    self.assertEqual(actual_model_response, _TEST_RESPONSE_TO_RECEIVE)
    self.assertTrue(actual_answer)

  @mock.patch('common.utils.strip_string')
  @mock.patch('common.modeling.FakeModel.generate')
  @mock.patch('common.utils.extract_first_code_block')
  def test_revise_fact_base(
      self,
      mock_extract_first_code_block: mock.Mock,
      mock_generate: mock.Mock,
      mock_strip_string: mock.Mock,
  ) -> None:
    mock_strip_string.return_value = _TEST_PROMPT_TO_SEND
    mock_generate.return_value = _TEST_RESPONSE_TO_RECEIVE
    mock_extract_first_code_block.return_value = _TEST_REVISED_STATEMENT
    actual_model_response, actual_answer = classify_relevance.revise_fact(
        response=_TEST_RESPONSE,
        atomic_fact=_TEST_STATEMENT,
        model=_TEST_MODEL,
        do_debug=False,
        max_retries=1,
    )
    mock_strip_string.assert_called_once()
    mock_generate.assert_called_once_with(_TEST_PROMPT_TO_SEND, do_debug=False)
    mock_extract_first_code_block.assert_called_once_with(
        _TEST_RESPONSE_TO_RECEIVE, ignore_language=True
    )
    self.assertEqual(actual_model_response, _TEST_RESPONSE_TO_RECEIVE)
    self.assertEqual(actual_answer, _TEST_REVISED_STATEMENT)

  @mock.patch('common.utils.strip_string')
  @mock.patch('common.modeling.FakeModel.generate')
  @mock.patch('common.utils.extract_first_code_block')
  def test_revise_fact_not_parseable(
      self,
      mock_extract_first_code_block: mock.Mock,
      mock_generate: mock.Mock,
      mock_strip_string: mock.Mock,
  ) -> None:
    mock_strip_string.return_value = _TEST_PROMPT_TO_SEND
    mock_generate.return_value = _TEST_RESPONSE_TO_RECEIVE
    mock_extract_first_code_block.return_value = ''
    actual_model_response, actual_answer = classify_relevance.revise_fact(
        response=_TEST_RESPONSE,
        atomic_fact=_TEST_STATEMENT,
        model=_TEST_MODEL,
        do_debug=False,
        max_retries=0,
    )
    mock_strip_string.assert_called_once()
    mock_generate.assert_called_once_with(_TEST_PROMPT_TO_SEND, do_debug=False)
    mock_extract_first_code_block.assert_called_once_with(
        _TEST_RESPONSE_TO_RECEIVE, ignore_language=True
    )
    self.assertEqual(actual_model_response, _TEST_RESPONSE_TO_RECEIVE)
    self.assertEqual(actual_answer, _TEST_STATEMENT)

  @mock.patch('eval.safe.classify_relevance.check_relevance')
  @mock.patch('eval.safe.classify_relevance.revise_fact')
  def test_main(
      self, mock_revise_fact: mock.Mock, mock_check_relevance: mock.Mock
  ) -> None:
    is_relevant = True
    mock_revise_fact.return_value = (
        _TEST_RESPONSE_TO_RECEIVE, _TEST_REVISED_STATEMENT
    )
    mock_check_relevance.return_value = _TEST_RESPONSE_TO_RECEIVE, is_relevant
    expected_model_responses = {
        'atomic_fact': _TEST_STATEMENT,
        'revised_fact': _TEST_RESPONSE_TO_RECEIVE,
        'is_relevant': _TEST_RESPONSE_TO_RECEIVE,
    }
    actual_is_relevant, actual_revised_fact, actual_model_responses = (
        classify_relevance.main(
            prompt=_TEST_PROMPT,
            response=_TEST_RESPONSE,
            atomic_fact=_TEST_STATEMENT,
            model=_TEST_MODEL,
        )
    )
    mock_revise_fact.assert_called_once_with(
        response=_TEST_RESPONSE, atomic_fact=_TEST_STATEMENT, model=_TEST_MODEL
    )
    mock_check_relevance.assert_called_once_with(
        prompt=_TEST_PROMPT,
        response=_TEST_RESPONSE,
        atomic_fact=_TEST_REVISED_STATEMENT,
        model=_TEST_MODEL,
    )
    self.assertEqual(actual_is_relevant, is_relevant)
    self.assertEqual(actual_revised_fact, _TEST_REVISED_STATEMENT)
    self.assertEqual(actual_model_responses, expected_model_responses)


if __name__ == '__main__':
  absltest.main()
