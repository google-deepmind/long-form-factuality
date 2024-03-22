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
"""Tests for search_augmented_factuality_eval.py.

Run command:
```
python -m eval.safe.search_augmented_factuality_eval_test
```
"""

import copy
from unittest import mock

from absl.testing import absltest

# pylint: disable=g-bad-import-order
from common import modeling
from eval.safe import rate_atomic_fact
from eval.safe import search_augmented_factuality_eval
# pylint: enable=g-bad-import-order

_TEST_STATEMENT = 'Here is a full sentence.'
_TEST_ATOMIC_FACT = 'This is an atomic fact from the full sentence.'
_TEST_SELF_CONTAINED = 'This is a self-contained version of the atomic fact.'
_TEST_PROMPT = 'This was the original prompt sent to the model.'
_TEST_RESPONSE = 'This was the full response from the model.'

_TEST_RATER = modeling.FakeModel()

_TEST_RELEVANCE_DATA = {'is_relevant': True}
_TEST_ANSWER = search_augmented_factuality_eval.SUPPORTED_LABEL
_TEST_FINAL_ANSWER = rate_atomic_fact.FinalAnswer(
    response='', answer=_TEST_ANSWER
)
_TEST_FINAL_ANSWER_AS_DICT = {'answer': _TEST_ANSWER}
_TEST_GOOGLE_SEARCH_DICT = {'google_searches': ['Some Google Searches']}

_TEST_CHECKED_STATEMENT = search_augmented_factuality_eval.CheckedStatement(
    sentence=_TEST_STATEMENT,
    atomic_fact=_TEST_ATOMIC_FACT,
    self_contained_atomic_fact=_TEST_ATOMIC_FACT,
    rate_data=_TEST_FINAL_ANSWER,
)


class SearchAugmentedFactualityEvalTest(absltest.TestCase):

  def test_checked_statement_init_no_rate_data_no_relevance_data(self) -> None:
    expected_data = {
        'sentence': _TEST_STATEMENT,
        'atomic_fact': _TEST_ATOMIC_FACT,
        'self_contained_atomic_fact': _TEST_ATOMIC_FACT,
        'relevance_data': None,
        'rate_data': None,
        'annotation': _TEST_ANSWER,
    }
    test_checked_statement = search_augmented_factuality_eval.CheckedStatement(
        sentence=_TEST_STATEMENT,
        atomic_fact=_TEST_ATOMIC_FACT,
        self_contained_atomic_fact=_TEST_ATOMIC_FACT,
        annotation=_TEST_ANSWER,
    )
    self.assertEqual(test_checked_statement.sentence, _TEST_STATEMENT)
    self.assertEqual(test_checked_statement.atomic_fact, _TEST_ATOMIC_FACT)
    self.assertEqual(
        test_checked_statement.self_contained_atomic_fact, _TEST_ATOMIC_FACT
    )
    self.assertIsNone(test_checked_statement.relevance_data)
    self.assertIsNone(test_checked_statement.rate_data)
    self.assertEqual(test_checked_statement.annotation, _TEST_ANSWER)
    self.assertEqual(test_checked_statement.data, expected_data)

  @mock.patch('dataclasses.asdict')
  def test_checked_statement_init_with_rate_data_with_relevant_data(
      self, mock_asdict: mock.Mock
  ) -> None:
    mock_asdict.return_value = _TEST_FINAL_ANSWER_AS_DICT
    expected_data = {
        'sentence': _TEST_STATEMENT,
        'atomic_fact': _TEST_ATOMIC_FACT,
        'self_contained_atomic_fact': _TEST_ATOMIC_FACT,
        'relevance_data': _TEST_RELEVANCE_DATA,
        'rate_data': _TEST_FINAL_ANSWER_AS_DICT,
        'annotation': _TEST_ANSWER,
    }
    test_checked_statement = search_augmented_factuality_eval.CheckedStatement(
        sentence=_TEST_STATEMENT,
        atomic_fact=_TEST_ATOMIC_FACT,
        self_contained_atomic_fact=_TEST_ATOMIC_FACT,
        relevance_data=_TEST_RELEVANCE_DATA,
        rate_data=_TEST_FINAL_ANSWER,
        annotation=_TEST_ANSWER,
    )
    self.assertEqual(test_checked_statement.sentence, _TEST_STATEMENT)
    self.assertEqual(test_checked_statement.atomic_fact, _TEST_ATOMIC_FACT)
    self.assertEqual(
        test_checked_statement.self_contained_atomic_fact, _TEST_ATOMIC_FACT
    )
    self.assertEqual(
        test_checked_statement.relevance_data, _TEST_RELEVANCE_DATA
    )
    self.assertEqual(test_checked_statement.rate_data, _TEST_FINAL_ANSWER)
    self.assertEqual(test_checked_statement.annotation, _TEST_ANSWER)
    mock_asdict.assert_called_once_with(_TEST_FINAL_ANSWER)
    self.assertEqual(test_checked_statement.data, expected_data)

  @mock.patch('common.utils.maybe_print_error')
  def test_count_labels(self, mock_maybe_print_error: mock.Mock) -> None:
    checked_supported = search_augmented_factuality_eval.CheckedStatement(
        sentence=_TEST_STATEMENT,
        atomic_fact=_TEST_ATOMIC_FACT,
        self_contained_atomic_fact=_TEST_ATOMIC_FACT,
        annotation=search_augmented_factuality_eval.SUPPORTED_LABEL,
    )
    checked_irrelevant = search_augmented_factuality_eval.CheckedStatement(
        sentence=_TEST_STATEMENT,
        atomic_fact=_TEST_ATOMIC_FACT,
        self_contained_atomic_fact=_TEST_ATOMIC_FACT,
        annotation=search_augmented_factuality_eval.IRRELEVANT_LABEL,
    )
    checked_unsupported = search_augmented_factuality_eval.CheckedStatement(
        sentence=_TEST_STATEMENT,
        atomic_fact=_TEST_ATOMIC_FACT,
        self_contained_atomic_fact=_TEST_ATOMIC_FACT,
        annotation=search_augmented_factuality_eval.NOT_SUPPORTED_LABEL,
    )
    checked_none = search_augmented_factuality_eval.CheckedStatement(
        sentence=_TEST_STATEMENT,
        atomic_fact=_TEST_ATOMIC_FACT,
        self_contained_atomic_fact=_TEST_ATOMIC_FACT,
    )
    num_supported, num_irrelevant, num_unsupported = 4, 3, 2
    expected_output = {
        search_augmented_factuality_eval.SUPPORTED_LABEL: num_supported,
        search_augmented_factuality_eval.IRRELEVANT_LABEL: num_irrelevant,
        search_augmented_factuality_eval.NOT_SUPPORTED_LABEL: num_unsupported,
    }
    statements = (
        [copy.deepcopy(checked_supported) for _ in range(num_supported)]
        + [copy.deepcopy(checked_irrelevant) for _ in range(num_irrelevant)]
        + [copy.deepcopy(checked_unsupported) for _ in range(num_unsupported)]
        + [None, checked_none]
    )
    actual_output = search_augmented_factuality_eval.count_labels(statements)
    self.assertEqual(actual_output, expected_output)
    mock_maybe_print_error.assert_not_called()

  @mock.patch('eval.safe.classify_relevance.main')
  @mock.patch('eval.safe.search_augmented_factuality_eval.CheckedStatement')
  def test_classify_relevance_and_rate_single_not_relevant(
      self,
      mock_checked_statement: mock.Mock,
      mock_classify_relevance: mock.Mock,
  ) -> None:
    mock_classify_relevance.return_value = (
        False, _TEST_SELF_CONTAINED, _TEST_RELEVANCE_DATA
    )
    mock_checked_statement.return_value = _TEST_CHECKED_STATEMENT
    actual_checked_statement, actual_revised_dict, actual_steps_dict = (
        search_augmented_factuality_eval.classify_relevance_and_rate_single(
            prompt=_TEST_PROMPT,
            response=_TEST_RESPONSE,
            sentence=_TEST_STATEMENT,
            atomic_fact=_TEST_ATOMIC_FACT,
            rater=_TEST_RATER,
        )
    )
    mock_classify_relevance.assert_called_once_with(
        _TEST_PROMPT,
        _TEST_RESPONSE,
        atomic_fact=_TEST_ATOMIC_FACT,
        model=_TEST_RATER,
    )
    mock_checked_statement.assert_called_once_with(
        sentence=_TEST_STATEMENT,
        atomic_fact=_TEST_ATOMIC_FACT,
        self_contained_atomic_fact=_TEST_SELF_CONTAINED,
        relevance_data=_TEST_RELEVANCE_DATA,
        annotation=search_augmented_factuality_eval.IRRELEVANT_LABEL,
    )
    self.assertEqual(actual_checked_statement, _TEST_CHECKED_STATEMENT)
    self.assertEqual(actual_revised_dict, _TEST_RELEVANCE_DATA)
    self.assertEqual(actual_steps_dict, {})

  @mock.patch('eval.safe.classify_relevance.main')
  @mock.patch('eval.safe.rate_atomic_fact.check_atomic_fact')
  def test_classify_relevance_and_rate_single_relevant_no_rate_data(
      self,
      mock_check_atomic_fact: mock.Mock,
      mock_classify_relevance: mock.Mock,
  ) -> None:
    mock_classify_relevance.return_value = (
        True, _TEST_SELF_CONTAINED, _TEST_RELEVANCE_DATA
    )
    mock_check_atomic_fact.return_value = None, {}
    self.assertRaises(
        ValueError,
        search_augmented_factuality_eval.classify_relevance_and_rate_single,
        prompt=_TEST_PROMPT,
        response=_TEST_RESPONSE,
        sentence=_TEST_STATEMENT,
        atomic_fact=_TEST_ATOMIC_FACT,
        rater=_TEST_RATER,
    )

  @mock.patch('eval.safe.classify_relevance.main')
  @mock.patch('eval.safe.rate_atomic_fact.check_atomic_fact')
  @mock.patch('eval.safe.search_augmented_factuality_eval.CheckedStatement')
  def test_classify_relevance_and_rate_single_relevant_with_rate_data(
      self,
      mock_checked_statement: mock.Mock,
      mock_check_atomic_fact: mock.Mock,
      mock_classify_relevance: mock.Mock,
  ) -> None:
    mock_classify_relevance.return_value = (
        True, _TEST_SELF_CONTAINED, _TEST_RELEVANCE_DATA
    )
    mock_check_atomic_fact.return_value = (
        _TEST_FINAL_ANSWER, _TEST_GOOGLE_SEARCH_DICT
    )
    mock_checked_statement.return_value = _TEST_CHECKED_STATEMENT
    actual_checked_statement, actual_revised_dict, actual_steps_dict = (
        search_augmented_factuality_eval.classify_relevance_and_rate_single(
            prompt=_TEST_PROMPT,
            response=_TEST_RESPONSE,
            sentence=_TEST_STATEMENT,
            atomic_fact=_TEST_ATOMIC_FACT,
            rater=_TEST_RATER,
        )
    )
    mock_classify_relevance.assert_called_once_with(
        _TEST_PROMPT,
        _TEST_RESPONSE,
        atomic_fact=_TEST_ATOMIC_FACT,
        model=_TEST_RATER,
    )
    mock_check_atomic_fact.assert_called_once_with(
        atomic_fact=_TEST_SELF_CONTAINED, rater=_TEST_RATER
    )
    mock_checked_statement.assert_called_once_with(
        sentence=_TEST_STATEMENT,
        atomic_fact=_TEST_ATOMIC_FACT,
        self_contained_atomic_fact=_TEST_SELF_CONTAINED,
        relevance_data=_TEST_RELEVANCE_DATA,
        rate_data=_TEST_FINAL_ANSWER,
        annotation=_TEST_ANSWER,
    )
    self.assertEqual(actual_checked_statement, _TEST_CHECKED_STATEMENT)
    self.assertEqual(actual_revised_dict, _TEST_RELEVANCE_DATA)
    self.assertEqual(actual_steps_dict, _TEST_GOOGLE_SEARCH_DICT)

  def test_classify_relevance_and_rate_no_atomic_facts(self) -> None:
    self.assertRaises(
        AssertionError,
        search_augmented_factuality_eval.classify_relevance_and_rate,
        prompt=_TEST_PROMPT,
        response=_TEST_RESPONSE,
        sentences_and_atomic_facts=[{'sentence': _TEST_STATEMENT}],
        rater=_TEST_RATER,
    )

  def test_classify_relevance_and_rate_invalid_atomic_facts(self) -> None:
    self.assertRaises(
        AssertionError,
        search_augmented_factuality_eval.classify_relevance_and_rate,
        prompt=_TEST_PROMPT,
        response=_TEST_RESPONSE,
        sentences_and_atomic_facts=[
            {'sentence': _TEST_STATEMENT, 'atomic_facts': _TEST_ATOMIC_FACT}
        ],
        rater=_TEST_RATER,
    )

  @mock.patch(
      'eval.safe.search_augmented_factuality_eval.'
      'classify_relevance_and_rate_single'
  )
  @mock.patch('eval.safe.search_augmented_factuality_eval.count_labels')
  def test_classify_relevance_and_rate_base(
      self,
      mock_count_labels: mock.Mock,
      mock_classify_relevance_and_rate_single: mock.Mock,
  ) -> None:
    mock_classify_relevance_and_rate_single.return_value = (
        _TEST_CHECKED_STATEMENT, _TEST_RELEVANCE_DATA, _TEST_GOOGLE_SEARCH_DICT
    )
    mock_count_labels.return_value = {
        search_augmented_factuality_eval.SUPPORTED_LABEL: 1
    }
    expected_output = {
        'checked_statements': [_TEST_CHECKED_STATEMENT.data],
        'revised_fact_jsonified_all': [_TEST_RELEVANCE_DATA],
        'past_steps_jsonified_all': [_TEST_GOOGLE_SEARCH_DICT],
        search_augmented_factuality_eval.SUPPORTED_LABEL: 1,
    }
    actual_output = (
        search_augmented_factuality_eval.classify_relevance_and_rate(
            prompt=_TEST_PROMPT,
            response=_TEST_RESPONSE,
            sentences_and_atomic_facts=[{
                'sentence': _TEST_STATEMENT, 'atomic_facts': [_TEST_ATOMIC_FACT]
            }],
            rater=_TEST_RATER,
        )
    )
    mock_classify_relevance_and_rate_single.assert_called_once_with(
        prompt=_TEST_PROMPT,
        response=_TEST_RESPONSE,
        sentence=_TEST_STATEMENT,
        atomic_fact=_TEST_ATOMIC_FACT,
        rater=_TEST_RATER,
    )
    mock_count_labels.assert_called_once_with(
        checked_statements=[_TEST_CHECKED_STATEMENT]
    )
    self.assertEqual(actual_output, expected_output)

  @mock.patch(
      'eval.safe.search_augmented_factuality_eval.'
      'classify_relevance_and_rate_single'
  )
  @mock.patch('eval.safe.search_augmented_factuality_eval.count_labels')
  @mock.patch('common.utils.maybe_print_error')
  def test_classify_relevance_and_rate_all_failures(
      self,
      mock_maybe_print_error: mock.Mock,
      mock_count_labels: mock.Mock,
      mock_classify_relevance_and_rate_single: mock.Mock,
  ) -> None:
    mock_classify_relevance_and_rate_single.side_effect = ValueError()
    mock_count_labels.return_value = {}
    expected_output = {
        'checked_statements': [],
        'revised_fact_jsonified_all': [],
        'past_steps_jsonified_all': [],
    }
    actual_output = (
        search_augmented_factuality_eval.classify_relevance_and_rate(
            prompt=_TEST_PROMPT,
            response=_TEST_RESPONSE,
            sentences_and_atomic_facts=[{
                'sentence': _TEST_STATEMENT, 'atomic_facts': [_TEST_ATOMIC_FACT]
            }],
            rater=_TEST_RATER,
        )
    )
    mock_classify_relevance_and_rate_single.assert_called()
    mock_count_labels.assert_called_once_with(checked_statements=[])
    mock_maybe_print_error.assert_called()
    self.assertEqual(actual_output, expected_output)

  @mock.patch('eval.safe.get_atomic_facts.main')
  @mock.patch(
      'eval.safe.search_augmented_factuality_eval.classify_relevance_and_rate'
  )
  def test_main(
      self,
      mock_classify_relevance_and_rate: mock.Mock,
      mock_get_atomic_facts: mock.Mock,
  ) -> None:
    sentences_and_atomic_facts = [
        {'sentence': _TEST_STATEMENT, 'atomic_facts': [_TEST_ATOMIC_FACT]}
    ]
    atomic_facts_dict = {'all_atomic_facts': sentences_and_atomic_facts}
    rating_result_dict = {
        'checked_statements': [_TEST_CHECKED_STATEMENT.data],
        'revised_fact_jsonified_all': [_TEST_RELEVANCE_DATA],
        'past_steps_jsonified_all': [_TEST_GOOGLE_SEARCH_DICT],
        search_augmented_factuality_eval.SUPPORTED_LABEL: 1,
    }
    mock_get_atomic_facts.return_value = atomic_facts_dict
    mock_classify_relevance_and_rate.return_value = rating_result_dict
    expected_output = {
        'prompt': _TEST_PROMPT,
        'response': _TEST_RESPONSE,
        **atomic_facts_dict,
        **rating_result_dict,
    }
    actual_output = search_augmented_factuality_eval.main(
        prompt=_TEST_PROMPT, response=_TEST_RESPONSE, rater=_TEST_RATER
    )
    mock_get_atomic_facts.assert_called_once_with(
        response=_TEST_RESPONSE, model=_TEST_RATER
    )
    mock_classify_relevance_and_rate.assert_called_once_with(
        prompt=_TEST_PROMPT,
        response=_TEST_RESPONSE,
        sentences_and_atomic_facts=sentences_and_atomic_facts,
        rater=_TEST_RATER,
    )
    self.assertEqual(actual_output, expected_output)

if __name__ == '__main__':
  absltest.main()
