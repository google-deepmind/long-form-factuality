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
"""Tests for correlation_vs_factscore.py.

Run command:
```
python -m eval.correlation_vs_factscore_test
```
"""

import copy
import math
from unittest import mock

from absl.testing import absltest
from scipy import stats

# pylint: disable=g-bad-import-order
from common import modeling
from eval import correlation_vs_factscore
from eval.safe import search_augmented_factuality_eval as safe
# pylint: enable=g-bad-import-order

_TEST_CORRELATION = 0.5
_TEST_P_VALUE = 0.005
_TEST_PEARSON = stats._stats_py.PearsonRResult(
    statistic=_TEST_CORRELATION,
    pvalue=_TEST_P_VALUE,
    alternative=None,
    n=None,
    x=None,
    y=None,
)
_TEST_SPEARMAN = stats._stats_py.SignificanceResult(
    statistic=_TEST_CORRELATION, pvalue=_TEST_P_VALUE
)

_TEST_SENTENCE = 'Lanny Flaherty is an American actor born on December 18, 1949'
_TEST_ATOMIC_FACTS = [
    {'text': 'Lanny Flaherty is an American.', 'label': 'S'},
    {'text': 'Lanny Flaherty is an actor.', 'label': 'IR'},
    {'text': 'Lanny Flaherty was born on December 18, 1949.', 'label': 'NS'},
]
_TEST_ANNOTATIONS = [
    {'text': _TEST_SENTENCE, 'human-atomic-facts': _TEST_ATOMIC_FACTS}
]
_TEST_LABEL_TO_COUNT = {label: 1 for label in ['S', 'IR', 'NS']}
_TEST_SENTENCE_TO_ATOMIC_FACT_AND_LABEL = {
    _TEST_SENTENCE: [
        {'atomic_fact': a['text'], 'label': a['label']}
        for a in _TEST_ATOMIC_FACTS
    ]
}

_TEST_TOPIC = 'Lanny Flaherty'
_TEST_PROMPT = f'Question: Tell me a bio of {_TEST_TOPIC}.'
_TEST_RESPONSE = _TEST_SENTENCE + '... response continues ...'
_TEST_MODEL = 'ChatGPT'

_TEST_FILEPATH = f'/some/file/path/{_TEST_MODEL}.jsonl'
_TEST_FACTSCORE_IN_DICT = {
    'input': _TEST_PROMPT,
    'output': _TEST_RESPONSE,
    'topic': _TEST_TOPIC,
    'annotations': _TEST_ANNOTATIONS,
}
_TEST_FACTSCORE_DATA = {
    'model_name': _TEST_MODEL,
    'prompt': _TEST_PROMPT,
    'response': _TEST_RESPONSE,
    'metrics': _TEST_LABEL_TO_COUNT,
    'sentence_to_atomic_fact_and_label': (
        _TEST_SENTENCE_TO_ATOMIC_FACT_AND_LABEL
    ),
}

_NUM_SENTENCES = 3
_EMPTY_DATA_DICT = {
    correlation_vs_factscore.RAW_STR: {},
    correlation_vs_factscore.AGGREGATED_STR: {},
}
_TEST_RESPONSE_LEVEL_DICT = {
    correlation_vs_factscore.FACTSCORE: (
        [copy.deepcopy(_EMPTY_DATA_DICT) for _ in range(_NUM_SENTENCES)]
    ),
    correlation_vs_factscore.EVAL_METHOD: (
        [copy.deepcopy(_EMPTY_DATA_DICT) for _ in range(_NUM_SENTENCES)]
    ),
}
_TEST_F1 = 0.5

_TEST_GET_ATOMIC_FACT_DATA = {'num_claims': _NUM_SENTENCES}
_TEST_RATE_FACT_DATA = {
    label: 1 for label in [
        safe.SUPPORTED_LABEL, safe.IRRELEVANT_LABEL, safe.NOT_SUPPORTED_LABEL
    ]
}
_TEST_ENTIRE_AUTORATER_DATA = copy.deepcopy(
    _TEST_GET_ATOMIC_FACT_DATA
) | copy.deepcopy(_TEST_RATE_FACT_DATA)

_TEST_RATER_MODEL = modeling.FakeModel()


class CorrelationVsFactscoreTest(absltest.TestCase):

  def test_correlation_result_init_pearson(self) -> None:
    expected_output = {
        'correlation': _TEST_CORRELATION, 'p_value': _TEST_P_VALUE
    }
    actual_output = correlation_vs_factscore.CorrelationResult(_TEST_PEARSON)
    self.assertEqual(expected_output, actual_output.result)

  def test_correlation_result_init_spearman(self) -> None:
    expected_output = {
        'correlation': _TEST_CORRELATION, 'p_value': _TEST_P_VALUE
    }
    actual_output = correlation_vs_factscore.CorrelationResult(_TEST_SPEARMAN)
    self.assertEqual(expected_output, actual_output.result)

  def test_correlation_result_init_none(self) -> None:
    actual_output = correlation_vs_factscore.CorrelationResult(None)
    for value in actual_output.result.values():
      self.assertTrue(math.isnan(value))

  def test_correlation_result_init_unsupported_type(self) -> None:
    self.assertRaises(
        ValueError,
        correlation_vs_factscore.CorrelationResult,
        'An unsupported string',
    )

  def test_get_atomic_annotations_base(self) -> None:
    actual_label_to_count, actual_sentence_to_atomic_fact_and_label = (
        correlation_vs_factscore.get_atomic_annotations(_TEST_ANNOTATIONS)
    )
    self.assertEqual(_TEST_LABEL_TO_COUNT, actual_label_to_count)
    self.assertEqual(
        _TEST_SENTENCE_TO_ATOMIC_FACT_AND_LABEL,
        actual_sentence_to_atomic_fact_and_label,
    )

  def test_get_atomic_annotations_unsupported_label(self) -> None:
    test_annotations = [
        {
            'text': 'A sentence',
            'human-atomic-facts': [
                {'text': 'An atomic fact', 'label': 'An unsupported label'}
            ]
        }
    ]
    self.assertRaises(
        AssertionError,
        correlation_vs_factscore.get_atomic_annotations,
        test_annotations,
    )

  @mock.patch('common.utils.read_from_jsonlines')
  @mock.patch('eval.correlation_vs_factscore.get_atomic_annotations')
  def test_load_factscore_data_base(
      self,
      mock_get_atomic_annotations: mock.Mock,
      mock_read_from_jsonlines: mock.Mock,
  ) -> None:
    mock_read_from_jsonlines.return_value = [_TEST_FACTSCORE_IN_DICT]
    mock_get_atomic_annotations.return_value = (
        _TEST_LABEL_TO_COUNT, _TEST_SENTENCE_TO_ATOMIC_FACT_AND_LABEL
    )
    actual_output = correlation_vs_factscore.load_factscore_data(_TEST_FILEPATH)
    mock_read_from_jsonlines.assert_called_once_with(_TEST_FILEPATH)
    mock_get_atomic_annotations.assert_called_once_with(
        annotations=_TEST_ANNOTATIONS
    )
    self.assertLen(actual_output, 1)
    self.assertEqual(actual_output[0], _TEST_FACTSCORE_DATA)

  @mock.patch('common.utils.read_from_jsonlines')
  @mock.patch('eval.correlation_vs_factscore.get_atomic_annotations')
  def test_load_factscore_data_no_annotations(
      self,
      mock_get_atomic_annotations: mock.Mock,
      mock_read_from_jsonlines: mock.Mock,
  ) -> None:
    mock_read_from_jsonlines.return_value = [{}]
    actual_output = correlation_vs_factscore.load_factscore_data(_TEST_FILEPATH)
    mock_read_from_jsonlines.assert_called_once_with(_TEST_FILEPATH)
    mock_get_atomic_annotations.assert_not_called()
    self.assertEmpty(actual_output)

  @mock.patch('common.utils.read_from_jsonlines')
  @mock.patch('eval.correlation_vs_factscore.get_atomic_annotations')
  def test_load_factscore_data_no_labels(
      self,
      mock_get_atomic_annotations: mock.Mock,
      mock_read_from_jsonlines: mock.Mock,
  ) -> None:
    mock_read_from_jsonlines.return_value = [_TEST_FACTSCORE_IN_DICT]
    empty_labels = {label: 0 for label in _TEST_LABEL_TO_COUNT.keys()}
    mock_get_atomic_annotations.return_value = (
        empty_labels, _TEST_SENTENCE_TO_ATOMIC_FACT_AND_LABEL
    )
    actual_output = correlation_vs_factscore.load_factscore_data(_TEST_FILEPATH)
    mock_read_from_jsonlines.assert_called_once_with(_TEST_FILEPATH)
    mock_get_atomic_annotations.assert_called_once_with(
        annotations=_TEST_ANNOTATIONS
    )
    self.assertEmpty(actual_output)

  @mock.patch('eval.metric_utils.calculate_metrics')
  def test_update_response_level_dict_factscore(
      self, mock_calculate_metrics: mock.Mock
  ) -> None:
    test_response_level_dict = copy.deepcopy(_TEST_RESPONSE_LEVEL_DICT)
    mock_calculate_metrics.return_value = _TEST_F1
    correlation_vs_factscore.update_response_level_dict(
        response_level_scores=test_response_level_dict,
        eval_method=correlation_vs_factscore.FACTSCORE,
        safe_step='',
        index=0,
        data=copy.deepcopy(_TEST_FACTSCORE_DATA),
    )
    expected_output = {
        correlation_vs_factscore.RAW_STR: {
            'supported': _TEST_LABEL_TO_COUNT['S'],
            'irrelevant': _TEST_LABEL_TO_COUNT['IR'],
            'not_supported': _TEST_LABEL_TO_COUNT['NS'],
            'num_claims': len(_TEST_LABEL_TO_COUNT.keys()),
        },
        correlation_vs_factscore.AGGREGATED_STR: {
            correlation_vs_factscore._F1_LABEL: _TEST_F1,
        }
    }
    mock_calculate_metrics.assert_called_once()
    self.assertLen(test_response_level_dict, 2)
    factscore_d = test_response_level_dict[correlation_vs_factscore.FACTSCORE]
    eval_d = test_response_level_dict[correlation_vs_factscore.EVAL_METHOD]
    self.assertLen(factscore_d, 3)
    self.assertLen(eval_d, 3)
    self.assertEqual(factscore_d[0], expected_output)
    self.assertEqual(eval_d[0], _EMPTY_DATA_DICT)
    self.assertEqual(factscore_d[1], _EMPTY_DATA_DICT)
    self.assertEqual(eval_d[1], _EMPTY_DATA_DICT)
    self.assertEqual(factscore_d[2], _EMPTY_DATA_DICT)
    self.assertEqual(eval_d[2], _EMPTY_DATA_DICT)

  def test_update_response_level_dict_eval_identify_facts(self) -> None:
    test_response_level_dict = copy.deepcopy(_TEST_RESPONSE_LEVEL_DICT)
    correlation_vs_factscore.update_response_level_dict(
        response_level_scores=test_response_level_dict,
        eval_method=correlation_vs_factscore.EVAL_METHOD,
        safe_step=correlation_vs_factscore.IDENTIFY_FACTS,
        index=0,
        data=copy.deepcopy(_TEST_GET_ATOMIC_FACT_DATA),
    )
    expected_output = {
        correlation_vs_factscore.RAW_STR: {
            'num_claims': _TEST_GET_ATOMIC_FACT_DATA['num_claims']
        },
        correlation_vs_factscore.AGGREGATED_STR: {},
    }
    self.assertLen(test_response_level_dict, 2)
    factscore_d = test_response_level_dict[correlation_vs_factscore.FACTSCORE]
    eval_d = test_response_level_dict[correlation_vs_factscore.EVAL_METHOD]
    self.assertLen(factscore_d, 3)
    self.assertLen(eval_d, 3)
    self.assertEqual(factscore_d[0], _EMPTY_DATA_DICT)
    self.assertEqual(eval_d[0], expected_output)
    self.assertEqual(factscore_d[1], _EMPTY_DATA_DICT)
    self.assertEqual(eval_d[1], _EMPTY_DATA_DICT)
    self.assertEqual(factscore_d[2], _EMPTY_DATA_DICT)
    self.assertEqual(eval_d[2], _EMPTY_DATA_DICT)

  @mock.patch('eval.metric_utils.calculate_metrics')
  def test_update_response_level_dict_eval_rate_facts(
      self, mock_calculate_metrics: mock.Mock
  ) -> None:
    test_response_level_dict = copy.deepcopy(_TEST_RESPONSE_LEVEL_DICT)
    mock_calculate_metrics.return_value = _TEST_F1
    correlation_vs_factscore.update_response_level_dict(
        response_level_scores=test_response_level_dict,
        eval_method=correlation_vs_factscore.EVAL_METHOD,
        safe_step=correlation_vs_factscore.RATE_FACTS,
        index=0,
        data=copy.deepcopy(_TEST_RATE_FACT_DATA),
    )
    expected_output = {
        correlation_vs_factscore.RAW_STR: {
            'supported': _TEST_RATE_FACT_DATA[safe.SUPPORTED_LABEL],
            'irrelevant': _TEST_RATE_FACT_DATA[safe.IRRELEVANT_LABEL],
            'not_supported': _TEST_RATE_FACT_DATA[safe.NOT_SUPPORTED_LABEL],
        },
        correlation_vs_factscore.AGGREGATED_STR: {
            correlation_vs_factscore._F1_LABEL: _TEST_F1,
        }
    }
    mock_calculate_metrics.assert_called_once()
    self.assertLen(test_response_level_dict, 2)
    factscore_d = test_response_level_dict[correlation_vs_factscore.FACTSCORE]
    eval_d = test_response_level_dict[correlation_vs_factscore.EVAL_METHOD]
    self.assertLen(factscore_d, 3)
    self.assertLen(eval_d, 3)
    self.assertEqual(factscore_d[0], _EMPTY_DATA_DICT)
    self.assertEqual(eval_d[0], expected_output)
    self.assertEqual(factscore_d[1], _EMPTY_DATA_DICT)
    self.assertEqual(eval_d[1], _EMPTY_DATA_DICT)
    self.assertEqual(factscore_d[2], _EMPTY_DATA_DICT)
    self.assertEqual(eval_d[2], _EMPTY_DATA_DICT)

  @mock.patch('eval.metric_utils.calculate_metrics')
  def test_update_response_level_dict_eval_entire_autorater(
      self, mock_calculate_metrics: mock.Mock
  ) -> None:
    test_response_level_dict = copy.deepcopy(_TEST_RESPONSE_LEVEL_DICT)
    mock_calculate_metrics.return_value = _TEST_F1
    correlation_vs_factscore.update_response_level_dict(
        response_level_scores=test_response_level_dict,
        eval_method=correlation_vs_factscore.EVAL_METHOD,
        safe_step=correlation_vs_factscore.ENTIRE_AUTORATER,
        index=0,
        data=copy.deepcopy(_TEST_ENTIRE_AUTORATER_DATA),
    )
    expected_output = {
        correlation_vs_factscore.RAW_STR: {
            'num_claims': _TEST_GET_ATOMIC_FACT_DATA['num_claims'],
            'supported': _TEST_RATE_FACT_DATA[safe.SUPPORTED_LABEL],
            'irrelevant': _TEST_RATE_FACT_DATA[safe.IRRELEVANT_LABEL],
            'not_supported': _TEST_RATE_FACT_DATA[safe.NOT_SUPPORTED_LABEL],
        },
        correlation_vs_factscore.AGGREGATED_STR: {
            correlation_vs_factscore._F1_LABEL: _TEST_F1,
        },
    }
    mock_calculate_metrics.assert_called_once()
    self.assertLen(test_response_level_dict, 2)
    factscore_d = test_response_level_dict[correlation_vs_factscore.FACTSCORE]
    eval_d = test_response_level_dict[correlation_vs_factscore.EVAL_METHOD]
    self.assertLen(factscore_d, 3)
    self.assertLen(eval_d, 3)
    self.assertEqual(factscore_d[0], _EMPTY_DATA_DICT)
    self.assertEqual(eval_d[0], expected_output)
    self.assertEqual(factscore_d[1], _EMPTY_DATA_DICT)
    self.assertEqual(eval_d[1], _EMPTY_DATA_DICT)
    self.assertEqual(factscore_d[2], _EMPTY_DATA_DICT)
    self.assertEqual(eval_d[2], _EMPTY_DATA_DICT)

  def test_update_response_level_dict_eval_invalid_step(self) -> None:
    self.assertRaises(
        ValueError,
        correlation_vs_factscore.update_response_level_dict,
        response_level_scores={},
        eval_method=correlation_vs_factscore.EVAL_METHOD,
        safe_step='Some invalid step',
        index=0,
        data={},
    )

  def test_update_response_level_dict_invalid_method(self) -> None:
    self.assertRaises(
        ValueError,
        correlation_vs_factscore.update_response_level_dict,
        response_level_scores={},
        eval_method='Some invalid method',
        safe_step='',
        index=0,
        data={},
    )

  def test_run_eval_method_invalid_safe_step(self) -> None:
    self.assertRaises(
        AssertionError,
        correlation_vs_factscore.run_eval_method,
        all_factscore_data=[_TEST_FACTSCORE_DATA],
        rater_model=_TEST_RATER_MODEL,
        safe_step='Some invalid step',
        eval_in_parallel=True,
        show_progress_bar=False,
    )

  @mock.patch('eval.safe.get_atomic_facts.main')
  @mock.patch('eval.correlation_vs_factscore.update_response_level_dict')
  def test_run_eval_method_identify_facts(
      self,
      mock_update_response_level_dict: mock.Mock,
      mock_get_atomic_facts: mock.Mock,
  ) -> None:
    mock_get_atomic_facts.return_value = _TEST_GET_ATOMIC_FACT_DATA
    expected_output = {
        **copy.deepcopy(_TEST_FACTSCORE_DATA),
        **copy.deepcopy(_TEST_GET_ATOMIC_FACT_DATA),
    }
    actual_per_response_data, actual_per_prompt_data = (
        correlation_vs_factscore.run_eval_method(
            all_factscore_data=[_TEST_FACTSCORE_DATA],
            rater_model=_TEST_RATER_MODEL,
            safe_step=correlation_vs_factscore.IDENTIFY_FACTS,
            eval_in_parallel=True,
            show_progress_bar=False,
        )
    )
    mock_update_response_level_dict.assert_called()
    mock_get_atomic_facts.assert_called_once_with(
        response=_TEST_RESPONSE, model=_TEST_RATER_MODEL
    )
    self.assertLen(actual_per_prompt_data, 1)
    self.assertEqual(actual_per_prompt_data[0], expected_output)
    self.assertLen(actual_per_response_data, 2)

    for key in [
        correlation_vs_factscore.FACTSCORE, correlation_vs_factscore.EVAL_METHOD
    ]:
      self.assertIn(key, actual_per_response_data)
      self.assertIsInstance(actual_per_response_data[key], list)
      self.assertLen(actual_per_response_data[key], 1)
      self.assertIsInstance(actual_per_response_data[key][0], dict)
      self.assertEqual(actual_per_response_data[key][0], _EMPTY_DATA_DICT)

  @mock.patch(
      'eval.safe.search_augmented_factuality_eval.classify_relevance_and_rate'
  )
  @mock.patch('eval.correlation_vs_factscore.update_response_level_dict')
  def test_run_eval_method_rate_facts(
      self,
      mock_update_response_level_dict: mock.Mock,
      mock_classify_relevance_and_rate: mock.Mock,
  ) -> None:
    mock_classify_relevance_and_rate.return_value = _TEST_RATE_FACT_DATA
    expected_output = {
        **copy.deepcopy(_TEST_FACTSCORE_DATA),
        **copy.deepcopy(_TEST_RATE_FACT_DATA),
    }
    actual_per_response_data, actual_per_prompt_data = (
        correlation_vs_factscore.run_eval_method(
            all_factscore_data=[_TEST_FACTSCORE_DATA],
            rater_model=_TEST_RATER_MODEL,
            safe_step=correlation_vs_factscore.RATE_FACTS,
            eval_in_parallel=True,
            show_progress_bar=False,
        )
    )
    mock_update_response_level_dict.assert_called()
    mock_classify_relevance_and_rate.assert_called_once()
    self.assertLen(actual_per_prompt_data, 1)
    self.assertEqual(actual_per_prompt_data[0], expected_output)
    self.assertLen(actual_per_response_data, 2)

    for key in [
        correlation_vs_factscore.FACTSCORE, correlation_vs_factscore.EVAL_METHOD
    ]:
      self.assertIn(key, actual_per_response_data)
      self.assertIsInstance(actual_per_response_data[key], list)
      self.assertLen(actual_per_response_data[key], 1)
      self.assertIsInstance(actual_per_response_data[key][0], dict)
      self.assertEqual(actual_per_response_data[key][0], _EMPTY_DATA_DICT)

  @mock.patch('eval.safe.search_augmented_factuality_eval.main')
  @mock.patch('eval.correlation_vs_factscore.update_response_level_dict')
  def test_run_eval_method_entire_autorater(
      self, mock_update_response_level_dict: mock.Mock, mock_main: mock.Mock
  ) -> None:
    mock_main.return_value = _TEST_ENTIRE_AUTORATER_DATA
    expected_output = {
        **copy.deepcopy(_TEST_FACTSCORE_DATA),
        **copy.deepcopy(_TEST_ENTIRE_AUTORATER_DATA),
    }
    actual_per_response_data, actual_per_prompt_data = (
        correlation_vs_factscore.run_eval_method(
            all_factscore_data=[_TEST_FACTSCORE_DATA],
            rater_model=_TEST_RATER_MODEL,
            safe_step=correlation_vs_factscore.ENTIRE_AUTORATER,
            eval_in_parallel=True,
            show_progress_bar=False,
        )
    )
    mock_update_response_level_dict.assert_called()
    mock_main.assert_called_once_with(
        prompt=_TEST_PROMPT, response=_TEST_RESPONSE, rater=_TEST_RATER_MODEL
    )
    self.assertLen(actual_per_prompt_data, 1)
    self.assertEqual(actual_per_prompt_data[0], expected_output)
    self.assertLen(actual_per_response_data, 2)

    for key in [
        correlation_vs_factscore.FACTSCORE, correlation_vs_factscore.EVAL_METHOD
    ]:
      self.assertIn(key, actual_per_response_data)
      self.assertIsInstance(actual_per_response_data[key], list)
      self.assertLen(actual_per_response_data[key], 1)
      self.assertIsInstance(actual_per_response_data[key][0], dict)
      self.assertEqual(actual_per_response_data[key][0], _EMPTY_DATA_DICT)

  @mock.patch('eval.safe.search_augmented_factuality_eval.main')
  @mock.patch('eval.correlation_vs_factscore.update_response_level_dict')
  @mock.patch('common.utils.print_progress')
  def test_run_eval_method_entire_autorater_no_parallelize(
      self,
      mock_print_progress: mock.Mock,
      mock_update_response_level_dict: mock.Mock,
      mock_main: mock.Mock,
  ) -> None:
    mock_main.return_value = _TEST_ENTIRE_AUTORATER_DATA
    expected_output = {
        **copy.deepcopy(_TEST_FACTSCORE_DATA),
        **copy.deepcopy(_TEST_ENTIRE_AUTORATER_DATA),
    }
    actual_per_response_data, actual_per_prompt_data = (
        correlation_vs_factscore.run_eval_method(
            all_factscore_data=[_TEST_FACTSCORE_DATA],
            rater_model=_TEST_RATER_MODEL,
            safe_step=correlation_vs_factscore.ENTIRE_AUTORATER,
            eval_in_parallel=False,
            show_progress_bar=False,
        )
    )
    mock_update_response_level_dict.assert_called()
    mock_print_progress.assert_called_once_with('Scoring', 1, 1)
    mock_main.assert_called_once_with(
        prompt=_TEST_PROMPT, response=_TEST_RESPONSE, rater=_TEST_RATER_MODEL
    )
    self.assertLen(actual_per_prompt_data, 1)
    self.assertEqual(actual_per_prompt_data[0], expected_output)
    self.assertLen(actual_per_response_data, 2)

    for key in [
        correlation_vs_factscore.FACTSCORE, correlation_vs_factscore.EVAL_METHOD
    ]:
      self.assertIn(key, actual_per_response_data)
      self.assertIsInstance(actual_per_response_data[key], list)
      self.assertLen(actual_per_response_data[key], 1)
      self.assertIsInstance(actual_per_response_data[key][0], dict)
      self.assertEqual(actual_per_response_data[key][0], _EMPTY_DATA_DICT)

  def test_find_metric(self) -> None:
    test_full_label_count = {
        correlation_vs_factscore.RAW_STR: copy.deepcopy(_TEST_LABEL_TO_COUNT),
        correlation_vs_factscore.AGGREGATED_STR: {},
    }
    self.assertEqual(
        -1, correlation_vs_factscore.find_metric(
            test_full_label_count, 'Some unknown metric'
        )
    )

    for metric_name in _TEST_LABEL_TO_COUNT.keys():
      self.assertEqual(
          1, correlation_vs_factscore.find_metric(
              test_full_label_count, metric_name
          ),
      )

    for metric_name in _TEST_LABEL_TO_COUNT.keys():
      self.assertEqual(
          correlation_vs_factscore.find_metric({}, metric_name), -1
      )

  def test_list_metrics(self) -> None:
    test_full_label_count = [{
        correlation_vs_factscore.RAW_STR: copy.deepcopy(_TEST_LABEL_TO_COUNT),
        correlation_vs_factscore.AGGREGATED_STR: {},
    }]
    expected_output = list(_TEST_LABEL_TO_COUNT.keys())
    actual_output = correlation_vs_factscore.list_metrics(test_full_label_count)
    self.assertLen(actual_output, len(expected_output))
    self.assertEqual(set(actual_output), set(expected_output))

  @mock.patch('common.utils.save_buffer')
  @mock.patch('common.utils.print_info')
  def test_scatter_plot(
      self, mock_info: mock.Mock, mock_save_buffer: mock.Mock
  ) -> None:
    correlation_vs_factscore.scatter_plot(x=[0, 1], y=[0, 1])
    mock_info.assert_called()
    mock_save_buffer.assert_called()

  @mock.patch('eval.correlation_vs_factscore.scatter_plot')
  @mock.patch('scipy.stats.pearsonr')
  @mock.patch('scipy.stats.spearmanr')
  def test_compute_correlation_base(
      self,
      mock_spearmanr: mock.Mock,
      mock_pearsonr: mock.Mock,
      mock_scatter_plot: mock.Mock,
  ) -> None:
    response_level_scores = {
        correlation_vs_factscore.FACTSCORE: [
            {correlation_vs_factscore.RAW_STR: {'S': 0, 'N': 1}},
            {correlation_vs_factscore.RAW_STR: {'S': 1, 'N': 2}},
        ],
        correlation_vs_factscore.EVAL_METHOD: [
            {correlation_vs_factscore.RAW_STR: {'S': 0, 'M': 1}},
            {correlation_vs_factscore.RAW_STR: {'S': 1, 'M': 2}},
        ],
    }
    mock_pearsonr.return_value = _TEST_PEARSON
    mock_spearmanr.return_value = _TEST_SPEARMAN
    expected_output = {
        'S': {
            each: {'correlation': _TEST_CORRELATION, 'p_value': _TEST_P_VALUE}
            for each in ['Pearson', 'Spearman']
        }
    }
    actual_output = correlation_vs_factscore.compute_correlation(
        response_level_scores=response_level_scores
    )
    mock_pearsonr.assert_called()
    mock_spearmanr.assert_called()
    mock_scatter_plot.assert_called()
    self.assertEqual(actual_output, expected_output)

  @mock.patch('eval.correlation_vs_factscore.scatter_plot')
  @mock.patch('scipy.stats.pearsonr')
  @mock.patch('scipy.stats.spearmanr')
  def test_compute_correlation_one_data_point(
      self,
      mock_spearmanr: mock.Mock,
      mock_pearsonr: mock.Mock,
      mock_scatter_plot: mock.Mock,
  ) -> None:
    response_level_scores = {
        correlation_vs_factscore.FACTSCORE: [
            {correlation_vs_factscore.RAW_STR: {'S': 0, 'N': 1}},
        ],
        correlation_vs_factscore.EVAL_METHOD: [
            {correlation_vs_factscore.RAW_STR: {'S': 0, 'M': 1}},
        ],
    }
    actual_output = correlation_vs_factscore.compute_correlation(
        response_level_scores=response_level_scores
    )
    mock_pearsonr.assert_not_called()
    mock_spearmanr.assert_not_called()
    mock_scatter_plot.assert_called()
    self.assertIsInstance(actual_output, dict)
    self.assertLen(actual_output, 1)
    self.assertIn('S', actual_output)
    self.assertIsInstance(actual_output['S'], dict)
    self.assertLen(actual_output['S'], 2)
    self.assertIn('Pearson', actual_output['S'])
    self.assertIsInstance(actual_output['S']['Pearson'], dict)
    self.assertIn('correlation', actual_output['S']['Pearson'])
    self.assertTrue(math.isnan(actual_output['S']['Pearson']['correlation']))
    self.assertIn('p_value', actual_output['S']['Pearson'])
    self.assertTrue(math.isnan(actual_output['S']['Pearson']['p_value']))
    self.assertIn('Spearman', actual_output['S'])
    self.assertIsInstance(actual_output['S']['Spearman'], dict)
    self.assertIn('correlation', actual_output['S']['Spearman'])
    self.assertTrue(math.isnan(actual_output['S']['Spearman']['correlation']))
    self.assertIn('p_value', actual_output['S']['Spearman'])
    self.assertTrue(math.isnan(actual_output['S']['Spearman']['p_value']))

  @mock.patch('scipy.stats.pearsonr')
  @mock.patch('scipy.stats.spearmanr')
  def test_compute_correlation_no_intersecting_metrics(
      self, mock_spearmanr: mock.Mock, mock_pearsonr: mock.Mock
  ) -> None:
    response_level_scores = {
        correlation_vs_factscore.FACTSCORE: [
            {correlation_vs_factscore.RAW_STR: {'N': 1}},
        ],
        correlation_vs_factscore.EVAL_METHOD: [
            {correlation_vs_factscore.RAW_STR: {'M': 1}},
        ],
    }
    actual_output = correlation_vs_factscore.compute_correlation(
        response_level_scores=response_level_scores
    )
    mock_pearsonr.assert_not_called()
    mock_spearmanr.assert_not_called()
    self.assertEmpty(actual_output)

  @mock.patch('builtins.round')
  @mock.patch('builtins.print')
  @mock.patch('eval.metric_utils.round_to_sigfigs')
  def test_print_correlation_results(
      self,
      mock_round_to_sigfigs: mock.Mock,
      mock_print: mock.Mock,
      mock_round: mock.Mock,
  ) -> None:
    mock_round.return_value = _TEST_CORRELATION
    mock_round_to_sigfigs.return_value = _TEST_P_VALUE
    correlation_scores = {
        'S': {
            each: {'correlation': _TEST_CORRELATION, 'p_value': _TEST_P_VALUE}
            for each in ['Pearson', 'Spearman']
        }
    }
    correlation_vs_factscore.print_correlation_results(correlation_scores)
    mock_round.assert_called()
    mock_round_to_sigfigs.assert_called()
    mock_print.assert_called()

  @mock.patch('common.utils.save_json')
  @mock.patch('common.utils.print_info')
  @mock.patch('common.utils.print_divider')
  @mock.patch('os.path.join')
  def test_save_results(
      self,
      mock_path_join: mock.Mock,
      mock_print_divider: mock.Mock,
      mock_print_info: mock.Mock,
      mock_save_json: mock.Mock,
  ) -> None:
    mock_path_join.return_value = _TEST_FILEPATH
    correlation_vs_factscore.save_results(_TEST_ENTIRE_AUTORATER_DATA)
    mock_save_json.assert_called_once_with(
        _TEST_FILEPATH, _TEST_ENTIRE_AUTORATER_DATA
    )
    mock_print_info.assert_called_once()
    mock_print_divider.assert_called_once_with()


if __name__ == '__main__':
  absltest.main()
