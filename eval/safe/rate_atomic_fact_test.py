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
"""Tests for rate_atomic_fact.py.

Run command:
```
python -m eval.safe.rate_atomic_fact_test
```
"""

from unittest import mock

from absl.testing import absltest

# pylint: disable=g-bad-import-order
from common import modeling
from eval.safe import rate_atomic_fact
# pylint: enable=g-bad-import-order

_TEST_SEARCH_QUERY = 'Here is a query to search.'
_TEST_SEARCH_RESULT = 'Here is the answer to the query.'
_TEST_API_KEY = '_TEST_API_KEY'
_TEST_POSTAMBLE = 'Here is a postamble for a query.'
_TEST_GOOGLE_SEARCH = rate_atomic_fact.GoogleSearchResult(
    query=_TEST_SEARCH_QUERY, result=_TEST_SEARCH_RESULT
)

_TEST_MODEL_INPUT = 'Here is a prompt to send to the model.'
_TEST_MODEL_RESPONSE = 'This is a response to a prompt.'
_TEST_MODEL = modeling.FakeModel()

_TEST_ANSWER = rate_atomic_fact.SUPPORTED_LABEL
_TEST_FINAL_ANSWER = rate_atomic_fact.FinalAnswer(
    response=_TEST_MODEL_RESPONSE, answer=_TEST_ANSWER
)

TEST_ATOMIC_FACT = 'Here is a self-contained atomic fact.'


class RateAtomicFactTest(absltest.TestCase):

  def test_google_search_result_init(self) -> None:
    self.assertEqual(_TEST_GOOGLE_SEARCH.query, _TEST_SEARCH_QUERY)
    self.assertEqual(_TEST_GOOGLE_SEARCH.result, _TEST_SEARCH_RESULT)

  def test_final_answer_init(self) -> None:
    self.assertEqual(_TEST_FINAL_ANSWER.response, _TEST_MODEL_RESPONSE)
    self.assertEqual(_TEST_FINAL_ANSWER.answer, _TEST_ANSWER)

  @mock.patch('eval.safe.query_serper.SerperAPI')
  def test_call_search_serper_no_postamble(
      self, mock_serper_api: mock.Mock
  ) -> None:
    serper_api = mock.Mock()
    serper_api.run.return_value = _TEST_SEARCH_RESULT
    mock_serper_api.return_value = serper_api
    actual_output = rate_atomic_fact.call_search(
        search_query=_TEST_SEARCH_QUERY,
        search_type='serper',
        num_searches=1,
        serper_api_key=_TEST_API_KEY,
        search_postamble='',
    )
    mock_serper_api.assert_called_once_with(_TEST_API_KEY, k=1)
    serper_api.run.assert_called_once_with(_TEST_SEARCH_QUERY, k=1)
    self.assertEqual(actual_output, _TEST_SEARCH_RESULT)

  @mock.patch('eval.safe.query_serper.SerperAPI')
  def test_call_search_serper_with_postamble(
      self, mock_serper_api: mock.Mock
  ) -> None:
    serper_api = mock.Mock()
    serper_api.run.return_value = _TEST_SEARCH_RESULT
    mock_serper_api.return_value = serper_api
    actual_output = rate_atomic_fact.call_search(
        search_query=_TEST_SEARCH_QUERY,
        search_type='serper',
        num_searches=1,
        serper_api_key=_TEST_API_KEY,
        search_postamble=_TEST_POSTAMBLE,
    )
    mock_serper_api.assert_called_once_with(_TEST_API_KEY, k=1)
    serper_api.run.assert_called_once_with(
        f'{_TEST_SEARCH_QUERY} {_TEST_POSTAMBLE}', k=1
    )
    self.assertEqual(actual_output, _TEST_SEARCH_RESULT)

  @mock.patch('common.utils.strip_string')
  @mock.patch('common.modeling.FakeModel.generate')
  @mock.patch('common.utils.extract_first_code_block')
  @mock.patch('eval.safe.rate_atomic_fact.GoogleSearchResult')
  @mock.patch('eval.safe.rate_atomic_fact.call_search')
  def test_maybe_get_next_search_valid_response(
      self,
      mock_call_search: mock.Mock,
      mock_google_search_result: mock.Mock,
      mock_extract_first_code_block: mock.Mock,
      mock_generate: mock.Mock,
      mock_strip_string: mock.Mock,
  ) -> None:
    mock_strip_string.return_value = _TEST_MODEL_INPUT
    mock_generate.return_value = _TEST_MODEL_RESPONSE
    mock_extract_first_code_block.return_value = _TEST_SEARCH_QUERY
    mock_call_search.return_value = _TEST_SEARCH_RESULT
    mock_google_search_result.return_value = _TEST_GOOGLE_SEARCH
    actual_output = rate_atomic_fact.maybe_get_next_search(
        atomic_fact=TEST_ATOMIC_FACT,
        past_searches=[_TEST_GOOGLE_SEARCH],
        model=_TEST_MODEL,
        debug=False,
    )
    mock_strip_string.assert_called_once()
    mock_generate.assert_called_once_with(_TEST_MODEL_INPUT, do_debug=False)
    mock_extract_first_code_block.assert_called_once_with(
        _TEST_MODEL_RESPONSE, ignore_language=True
    )
    mock_call_search.assert_called_once_with(_TEST_SEARCH_QUERY)
    mock_google_search_result.assert_called_once_with(
        query=_TEST_SEARCH_QUERY, result=_TEST_SEARCH_RESULT
    )
    self.assertEqual(actual_output, _TEST_GOOGLE_SEARCH)

  @mock.patch('common.utils.strip_string')
  @mock.patch('common.modeling.FakeModel.generate')
  @mock.patch('common.utils.extract_first_code_block')
  def test_maybe_get_next_search_unparseable_response(
      self,
      mock_extract_first_code_block: mock.Mock,
      mock_generate: mock.Mock,
      mock_strip_string: mock.Mock,
  ) -> None:
    mock_strip_string.return_value = _TEST_MODEL_INPUT
    mock_generate.return_value = _TEST_MODEL_RESPONSE
    mock_extract_first_code_block.return_value = ''
    actual_output = rate_atomic_fact.maybe_get_next_search(
        atomic_fact=TEST_ATOMIC_FACT,
        past_searches=[_TEST_GOOGLE_SEARCH],
        model=_TEST_MODEL,
        debug=False,
    )
    mock_strip_string.assert_called_once()
    mock_generate.assert_called_once_with(_TEST_MODEL_INPUT, do_debug=False)
    mock_extract_first_code_block.assert_called_once_with(
        _TEST_MODEL_RESPONSE, ignore_language=True
    )
    self.assertIsNone(actual_output)

  @mock.patch('common.utils.strip_string')
  @mock.patch('common.modeling.FakeModel.generate')
  @mock.patch('common.utils.extract_first_square_brackets')
  @mock.patch('re.sub')
  @mock.patch('eval.safe.rate_atomic_fact.FinalAnswer')
  def test_maybe_get_final_answer_valid_response(
      self,
      mock_final_answer: mock.Mock,
      mock_re_sub: mock.Mock,
      mock_extract_first_square_brackets: mock.Mock,
      mock_generate: mock.Mock,
      mock_strip_string: mock.Mock,
  ) -> None:
    mock_strip_string.return_value = _TEST_MODEL_INPUT
    mock_generate.return_value = _TEST_MODEL_RESPONSE
    mock_extract_first_square_brackets.return_value = _TEST_ANSWER
    mock_re_sub.return_value = _TEST_ANSWER
    mock_final_answer.return_value = _TEST_FINAL_ANSWER
    actual_output = rate_atomic_fact.maybe_get_final_answer(
        atomic_fact=TEST_ATOMIC_FACT,
        searches=[_TEST_GOOGLE_SEARCH],
        model=_TEST_MODEL,
        debug=False,
    )
    mock_strip_string.assert_called_once()
    mock_generate.assert_called_once_with(_TEST_MODEL_INPUT, do_debug=False)
    mock_extract_first_square_brackets.assert_called_once_with(
        _TEST_MODEL_RESPONSE
    )
    mock_re_sub.assert_called_once()
    mock_final_answer.assert_called_once_with(
        response=_TEST_MODEL_RESPONSE, answer=_TEST_ANSWER
    )
    self.assertEqual(actual_output, _TEST_FINAL_ANSWER)

  @mock.patch('common.utils.strip_string')
  @mock.patch('common.modeling.FakeModel.generate')
  @mock.patch('common.utils.extract_first_square_brackets')
  @mock.patch('re.sub')
  def test_maybe_get_final_answer_unparseable_response(
      self,
      mock_re_sub: mock.Mock,
      mock_extract_first_square_brackets: mock.Mock,
      mock_generate: mock.Mock,
      mock_strip_string: mock.Mock,
  ) -> None:
    mock_strip_string.return_value = _TEST_MODEL_INPUT
    mock_generate.return_value = _TEST_MODEL_RESPONSE
    mock_extract_first_square_brackets.return_value = _TEST_ANSWER
    mock_re_sub.return_value = 'Some unknown label'
    actual_output = rate_atomic_fact.maybe_get_final_answer(
        atomic_fact=TEST_ATOMIC_FACT,
        searches=[_TEST_GOOGLE_SEARCH],
        model=_TEST_MODEL,
        debug=False,
    )
    mock_strip_string.assert_called_once()
    mock_generate.assert_called_once_with(_TEST_MODEL_INPUT, do_debug=False)
    mock_extract_first_square_brackets.assert_called_once_with(
        _TEST_MODEL_RESPONSE
    )
    mock_re_sub.assert_called_once()
    self.assertIsNone(actual_output)

  @mock.patch('eval.safe.rate_atomic_fact.maybe_get_next_search')
  @mock.patch('eval.safe.rate_atomic_fact.maybe_get_final_answer')
  @mock.patch('common.utils.maybe_print_error')
  @mock.patch('dataclasses.asdict')
  def test_check_atomic_fact_base(
      self,
      mock_as_dict: mock.Mock,
      mock_maybe_print_error: mock.Mock,
      mock_maybe_get_final_answer: mock.Mock,
      mock_maybe_get_next_search: mock.Mock,
  ) -> None:
    expected_dict = {'query': _TEST_SEARCH_QUERY, 'result': _TEST_SEARCH_RESULT}
    mock_maybe_get_next_search.return_value = _TEST_GOOGLE_SEARCH
    mock_maybe_get_final_answer.return_value = _TEST_FINAL_ANSWER
    mock_as_dict.return_value = expected_dict
    expected_search_dicts = {'google_searches': [expected_dict]}
    actual_final_answer, actual_search_dicts = (
        rate_atomic_fact.check_atomic_fact(
            atomic_fact=TEST_ATOMIC_FACT,
            rater=_TEST_MODEL,
            max_steps=1,
            max_retries=0,
            debug=False,
        )
    )
    mock_maybe_get_next_search.assert_called_once()
    mock_maybe_get_final_answer.assert_called_once()
    mock_maybe_print_error.assert_not_called()
    self.assertEqual(actual_final_answer, _TEST_FINAL_ANSWER)
    self.assertEqual(actual_search_dicts, expected_search_dicts)

  @mock.patch('eval.safe.rate_atomic_fact.maybe_get_next_search')
  @mock.patch('eval.safe.rate_atomic_fact.maybe_get_final_answer')
  @mock.patch('common.utils.maybe_print_error')
  def test_check_atomic_fact_unsuccessful_parsing(
      self,
      mock_maybe_print_error: mock.Mock,
      mock_maybe_get_final_answer: mock.Mock,
      mock_maybe_get_next_search: mock.Mock,
  ) -> None:
    mock_maybe_get_next_search.return_value = None
    mock_maybe_get_final_answer.return_value = None
    expected_search_dicts = {'google_searches': []}
    actual_final_answer, actual_search_dicts = (
        rate_atomic_fact.check_atomic_fact(
            atomic_fact=TEST_ATOMIC_FACT,
            rater=_TEST_MODEL,
            max_steps=5,
            max_retries=0,
            debug=False,
        )
    )
    mock_maybe_get_next_search.assert_called_once()
    mock_maybe_get_final_answer.assert_called_once()
    mock_maybe_print_error.assert_called()
    self.assertIsNone(actual_final_answer)
    self.assertEqual(actual_search_dicts, expected_search_dicts)


if __name__ == '__main__':
  absltest.main()
