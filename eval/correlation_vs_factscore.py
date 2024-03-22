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
r"""Calculates correlation between SAFE and human ratings from FActScore.

Flags `eval_identify_atomic_facts_ablated`, `eval_rate_atomic_facts_ablated`,
and `eval_entire_safe` are separate, and thus control whether to run
each mutually-independent evaluation experiment.

Run command:
```
python -m eval.correlation_vs_factscore \
    --samples=100 \
    --eval_in_parallel=True \
    --save_results=True \
    --eval_identify_atomic_facts_ablated=False \
    --eval_rate_atomic_facts_ablated=False \
    --eval_entire_safe=False
```
"""

import datetime
import io
import os
import time
from typing import Any, Literal

from absl import app
from absl import flags
import langfun as lf
from matplotlib import pyplot as plt
from scipy import stats

# pylint: disable=g-bad-import-order
from common import modeling
from common import shared_config
from common import utils
from eval import metric_utils
from eval.safe import config as safe_config
from eval.safe import get_atomic_facts
from eval.safe import search_augmented_factuality_eval as safe
# pylint: enable=g-bad-import-order

_SAMPLES = flags.DEFINE_integer(
    'samples', default=-1, help='Number of samples to eval.'
)
_EVAL_IN_PARALLEL = flags.DEFINE_boolean(
    'eval_in_parallel', default=True, help='Whether to evaluate in parallel.'
)
_SAVE_RESULTS = flags.DEFINE_boolean(
    'save_results', default=True, help='Whether to save all results to a JSON.'
)
_EVAL_IDENTIFY_ATOMIC_FACTS_ABLATED = flags.DEFINE_bool(
    'eval_identify_atomic_facts_ablated',
    default=False,
    help='Whether to independently eval the method of getting atomic facts.',
)
_EVAL_RATE_ATOMIC_FACTS_ABLATED = flags.DEFINE_bool(
    'eval_rate_atomic_facts_ablated',
    default=False,
    help='Whether to independently eval the method of rating atomic facts.',
)
_EVAL_ENTIRE_SAFE = flags.DEFINE_bool(
    'eval_entire_safe',
    default=False,
    help='Whether to evaluate the entire SAFE pipeline together.',
)

_FACTSCORE_DATA_FOLDER = os.path.join(
    shared_config.root_dir, 'third_party/factscore/labeled_data/'
)
_DATE_AND_TIME = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
_HUMAN_ATOMIC_FACTS = 'human-atomic-facts'
_ATOMIC_FACT = 'atomic_fact'
_ATOMIC_FACTS = 'atomic_facts'
_SENTENCE = 'sentence'
_ANNOTATIONS = 'annotations'
_MODEL_NAME = 'model_name'
_INPUT = 'input'
_OUTPUT = 'output'
_PROMPT = 'prompt'
_RESPONSE = 'response'
_METRICS = 'metrics'
_SENTENCE_TO_ATOMIC_FACT_AND_LABEL = 'sentence_to_atomic_fact_and_label'
_CORRELATION = 'correlation'
_PEARSON = 'Pearson'
_SPEARMAN = 'Spearman'
_P_VALUE = 'p_value'
_TEXT = 'text'
_LABEL = 'label'
_S_LABEL = 'S'
_S_LABEL_FULL = 'supported'
_IR_LABEL = 'IR'
_IR_LABEL_FULL = 'irrelevant'
_NS_LABEL = 'NS'
_NS_LABEL_FULL = 'not_supported'
_C_LABEL = 'num_claims'
_F1_LABEL = f'f1_{metric_utils.DEFAULT_MAX_CLAIMS}'

EVAL_METHOD = 'search_augmented_factuality_eval'
FACTSCORE = 'factscore'
IDENTIFY_FACTS = 'identify_facts'
RATE_FACTS = 'rate_facts'
ENTIRE_AUTORATER = 'entire_safe'
RAW_STR = 'raw'
AGGREGATED_STR = 'aggregated'


class CorrelationResult:
  """A class to hold a correlation result."""

  def __init__(self, correlation_result: Any) -> None:
    self.result = {}

    if isinstance(
        correlation_result,
        stats._stats_py.SignificanceResult | stats._stats_py.PearsonRResult,
    ):
      self.result[_CORRELATION] = correlation_result.statistic
      self.result[_P_VALUE] = correlation_result.pvalue
    elif correlation_result is None:
      self.result[_CORRELATION] = float('nan')
      self.result[_P_VALUE] = float('nan')
    else:
      raise ValueError(f'Unknown correlation type: {type(correlation_result)}')


def get_atomic_annotations(
    annotations: list[dict[str, Any]]
) -> tuple[dict[str, int], dict[str, list[dict[str, Any]]]]:
  """Gets atomic annotations from the given annotations.

  Args:
    annotations: The list of annotation data for each prompt--response pair.

  Returns:
    label_to_count: The number of atomic facts annotated as each label.
    sentence_to_atomic_fact_and_label: A dictionary of a sentence to the atomic
      facts contained in that sentence.
  """
  label_to_count = {label: 0 for label in [_S_LABEL, _IR_LABEL, _NS_LABEL]}
  sentence_to_atomic_fact_and_label = {}

  for annotation in annotations:
    sentence = annotation[_TEXT]

    if _HUMAN_ATOMIC_FACTS in annotation and annotation[_HUMAN_ATOMIC_FACTS]:
      for atomic_annotation in annotation[_HUMAN_ATOMIC_FACTS]:
        label = atomic_annotation[_LABEL]
        assert label in label_to_count, f'Unknown label: {label}'
        label_to_count[label] += 1

        if sentence not in sentence_to_atomic_fact_and_label:
          sentence_to_atomic_fact_and_label[sentence] = []

        sentence_to_atomic_fact_and_label[sentence].append({
            _ATOMIC_FACT: atomic_annotation[_TEXT], _LABEL: label
        })

  return label_to_count, sentence_to_atomic_fact_and_label


def load_factscore_data(input_path: str) -> list[dict[str, Any]]:
  """Loads a single FActScore data file.

  Args:
    input_path: The path to the FActScore data file that should be loaded.

  Returns:
    result: A list of the raw data for each prompt-response pair in the file.
  """
  result = []

  for prompt_data in utils.read_from_jsonlines(input_path):
    if _ANNOTATIONS not in prompt_data or not prompt_data[_ANNOTATIONS]:
      continue  # punted responses are not annotated

    label_to_count, sentence_to_atomic_fact_and_label = get_atomic_annotations(
        annotations=prompt_data[_ANNOTATIONS]
    )

    # Do not include items without any ratings
    if not any(label_to_count.values()):
      continue

    result.append({
        _MODEL_NAME: os.path.basename(input_path).split('.')[0],
        _PROMPT: prompt_data[_INPUT],
        _RESPONSE: prompt_data[_OUTPUT],
        _METRICS: label_to_count,
        _SENTENCE_TO_ATOMIC_FACT_AND_LABEL: sentence_to_atomic_fact_and_label,
    })

  return result


def update_response_level_dict(
    response_level_scores: dict[str, list[dict[str, dict[Any, Any]]]],
    eval_method: Literal[FACTSCORE, EVAL_METHOD],
    safe_step: Literal[IDENTIFY_FACTS, RATE_FACTS, ENTIRE_AUTORATER],
    index: int,
    data: dict[str, Any],
) -> None:
  """Updates the response level dict with the given data."""
  if eval_method == FACTSCORE:
    num_claims = sum([int(v) for v in list(data[_METRICS].values())])
    response_level_scores[FACTSCORE][index][RAW_STR].update({
        _S_LABEL_FULL: data[_METRICS][_S_LABEL],
        _IR_LABEL_FULL: data[_METRICS][_IR_LABEL],
        _NS_LABEL_FULL: data[_METRICS][_NS_LABEL],
        _C_LABEL: num_claims,
    })
    response_level_scores[FACTSCORE][index][AGGREGATED_STR].update({
        _F1_LABEL: metric_utils.calculate_metrics(
            supported=data[_METRICS][_S_LABEL],
            not_supported=data[_METRICS][_NS_LABEL],
        )
    })
  elif eval_method == EVAL_METHOD:
    if safe_step == IDENTIFY_FACTS:
      response_level_scores[EVAL_METHOD][index][RAW_STR].update({
          _C_LABEL: data[_C_LABEL],
      })
    elif safe_step == RATE_FACTS:
      response_level_scores[EVAL_METHOD][index][RAW_STR].update({
          _S_LABEL_FULL: data[safe.SUPPORTED_LABEL],
          _IR_LABEL_FULL: data[safe.IRRELEVANT_LABEL],
          _NS_LABEL_FULL: data[safe.NOT_SUPPORTED_LABEL],
      })
      response_level_scores[EVAL_METHOD][index][AGGREGATED_STR].update({
          _F1_LABEL: metric_utils.calculate_metrics(
              supported=data[safe.SUPPORTED_LABEL],
              not_supported=data[safe.NOT_SUPPORTED_LABEL],
          )
      })
    elif safe_step == ENTIRE_AUTORATER:
      update_response_level_dict(
          response_level_scores, EVAL_METHOD, IDENTIFY_FACTS, index, data
      )
      update_response_level_dict(
          response_level_scores, EVAL_METHOD, RATE_FACTS, index, data
      )
    else:
      raise ValueError(f'Unknown safe_step: {safe_step}')
  else:
    raise ValueError(f'Unknown eval_method: {eval_method}')


def run_eval_method(
    all_factscore_data: list[dict[str, Any]],
    rater_model: modeling.Model,
    safe_step: Literal[IDENTIFY_FACTS, RATE_FACTS, ENTIRE_AUTORATER],
    eval_in_parallel: bool = False,
    show_progress_bar: bool = True,
) -> tuple[dict[str, list[dict[str, dict[str, Any]]]], list[dict[str, Any]]]:
  """Runs the given evaluation method on the FactScore data.

  Args:
    all_factscore_data: The list of the raw data for each prompt-response pair
      to evaluate.
    rater_model: The language model to use for SAFE.
    safe_step: The steps in the SAFE pipeline that should be tested.
    eval_in_parallel: Whether to parallelize eval across prompt-response pairs.
    show_progress_bar: Whether to show a progress bar if running evaluation in
      parallel.

  Returns:
    per_response_data: The metrics from FActScore and from SAFE for all
      prompt-response pairs.
    per_prompt_data: A combined list of the raw data and SAFE data for all
      prompt-response pairs, which is useful for debugging.
  """
  def get_atomic_facts_wrapped(
      factscore_data_and_index: tuple[dict[str, Any], int]
  ) -> dict[str, Any]:
    response = factscore_data_and_index[0][_RESPONSE]
    return get_atomic_facts.main(response=response, model=rater_model)

  def rate_atomic_facts_wrapped(
      factscore_data_and_index: tuple[dict[str, Any], int]
  ) -> dict[str, Any]:
    factscore_data = factscore_data_and_index[0]
    sentences_and_atomic_facts = []

    for sent, v in factscore_data[_SENTENCE_TO_ATOMIC_FACT_AND_LABEL].items():
      sentences_and_atomic_facts.append({
          _SENTENCE: sent, _ATOMIC_FACTS: [item[_ATOMIC_FACT] for item in v]
      })

    return safe.classify_relevance_and_rate(
        prompt=factscore_data[_PROMPT],
        response=factscore_data[_RESPONSE],
        sentences_and_atomic_facts=sentences_and_atomic_facts,
        rater=rater_model,
    )

  def eval_pipeline_wrapped(
      factscore_data_and_index: tuple[dict[str, Any], int]
  ) -> dict[str, Any]:
    factscore_data = factscore_data_and_index[0]
    return safe.main(
        prompt=factscore_data[_PROMPT],
        response=factscore_data[_RESPONSE],
        rater=rater_model,
    )

  step_to_func = {
      IDENTIFY_FACTS: get_atomic_facts_wrapped,
      RATE_FACTS: rate_atomic_facts_wrapped,
      ENTIRE_AUTORATER: eval_pipeline_wrapped,
  }
  assert safe_step in step_to_func, f'Unknown step: {safe_step}'
  func_to_eval = step_to_func[safe_step]

  num_prompts = len(all_factscore_data)
  per_prompt_data = [{'': None} for _ in range(num_prompts)]
  per_response_data = {
      FACTSCORE: [
          {RAW_STR: {}, AGGREGATED_STR: {}} for _ in range(num_prompts)
      ],
      EVAL_METHOD: [
          {RAW_STR: {}, AGGREGATED_STR: {}} for _ in range(num_prompts)
      ],
  }

  if eval_in_parallel:
    # We don't needed `ordered=True` because we're indexing the result objects.
    for factscore_data_and_index, evaluated_data, error in lf.concurrent_map(
        func_to_eval,
        [(item, index) for index, item in enumerate(all_factscore_data)],
        max_workers=30,
        show_progress=show_progress_bar,
    ):
      if error or not evaluated_data:
        utils.maybe_print_error(error)
      else:
        factscore_data, i = factscore_data_and_index
        update_response_level_dict(
            per_response_data, FACTSCORE, safe_step, i, factscore_data
        )
        update_response_level_dict(
            per_response_data, EVAL_METHOD, safe_step, i, evaluated_data
        )
        per_prompt_data[i] = {**factscore_data, **evaluated_data}
  else:
    completed = 0

    for i, factscore_data in enumerate(all_factscore_data):
      evaluated_data = func_to_eval((factscore_data, i))
      update_response_level_dict(
          per_response_data, FACTSCORE, safe_step, i, factscore_data
      )
      update_response_level_dict(
          per_response_data, EVAL_METHOD, safe_step, i, evaluated_data
      )
      per_prompt_data[i] = {**factscore_data, **evaluated_data}
      completed += 1
      utils.print_progress('Scoring', completed, len(all_factscore_data))

  return per_response_data, per_prompt_data


def find_metric(in_dict: dict[str, dict[str, Any]], metric_name: str) -> float:
  """Finds the given metric's value in the given metric dictionary."""
  if RAW_STR in in_dict and metric_name in in_dict[RAW_STR]:
    return in_dict[RAW_STR][metric_name]
  elif AGGREGATED_STR in in_dict and metric_name in in_dict[AGGREGATED_STR]:
    return in_dict[AGGREGATED_STR][metric_name]
  else:  # metric was not found
    return -1


def list_metrics(metric_dicts: list[dict[str, dict[str, Any]]]) -> list[str]:
  """lists all metrics in the given metric dictionary."""
  result = []

  for metric_dict in metric_dicts:
    for metric_type in [RAW_STR, AGGREGATED_STR]:
      if metric_type in metric_dict:
        result += list(metric_dict[metric_type].keys())

  return list(set(result))


def scatter_plot(
    x: list[int | float],
    y: list[int | float],
    title: str = '',
    x_axis_label: str = '',
    y_axis_label: str = '',
) -> None:
  """Shows a basic scatter plot of the given data."""
  _, ax = plt.subplots()
  ax.scatter(x, y)
  ax.set_title(title)
  ax.set_xlabel(x_axis_label)
  ax.set_ylabel(y_axis_label)
  out_folder = os.path.join(
      '/'.join(shared_config.path_to_result.split('/')[:-1]), 'figures'
  )
  out_name = f'{_DATE_AND_TIME}-{title}-x={x_axis_label}-y={y_axis_label}.png'
  out_path = os.path.join(out_folder, out_name)
  buffer = io.BytesIO()
  plt.savefig(buffer, format='png', dpi=500)
  utils.save_buffer(buffer, out_path)
  utils.print_info(f'Saved plot to:\n{out_path}', add_punctuation=False)


def compute_correlation(
    response_level_scores: dict[str, list[dict[str, dict[str, Any]]]],
    name: str = '',
) -> dict[str, dict[str, dict[str, float]]]:
  """Computes correlations between FactScore and the evaluation method.

  Args:
    response_level_scores: The metrics from FActScore and from SAFE for all
      prompt-response pairs.
    name: The name of the steps of SAFE that the scores were obtained using.

  Returns:
    result: A dictionary of each metric and the Pearson/Spearman correlations
      between FActScore and SAFE for that metric.
  """
  factscore_metrics = set(list_metrics(response_level_scores[FACTSCORE]))
  pred_metrics = set(list_metrics(response_level_scores[EVAL_METHOD]))
  result = {}

  for metric in factscore_metrics.intersection(pred_metrics):
    factscore_scores = [
        find_metric(response_level_scores[FACTSCORE][i], metric)
        for i in range(len(response_level_scores[FACTSCORE]))
    ]
    pred_scores = [
        find_metric(response_level_scores[EVAL_METHOD][i], metric)
        for i in range(len(response_level_scores[EVAL_METHOD]))
    ]

    # Remove failed runs
    for i in range(len(factscore_scores) - 1, -1, -1):
      if factscore_scores[i] == -1 or pred_scores[i] == -1:
        factscore_scores.pop(i)
        pred_scores.pop(i)

    # Cannot calculate correlation with only one data point
    if len(factscore_scores) <= 1 or len(pred_scores) <= 1:
      pearson_result, spearman_result = None, None
    else:
      pearson_result = stats.pearsonr(factscore_scores, pred_scores)
      spearman_result = stats.spearmanr(factscore_scores, pred_scores)

    scatter_plot(
        factscore_scores,
        pred_scores,
        f'{name}-{metric}',
        FACTSCORE,
        EVAL_METHOD,
    )
    result[metric] = {
        _PEARSON: CorrelationResult(pearson_result).result,
        _SPEARMAN: CorrelationResult(spearman_result).result,
    }

  return result


def print_correlation_results(
    correlation_scores: dict[str, dict[str, dict[str, float]]]
) -> None:
  for metric, correlations in correlation_scores.items():
    print(f'Correlations scores for `{metric}`')

    for correlation_type, score in correlations.items():
      rounded_correlation = round(score[_CORRELATION], 3)
      p_value = metric_utils.round_to_sigfigs(score[_P_VALUE], 3)
      print(f'    {correlation_type}: {rounded_correlation}, p={p_value}')


def save_results(
    results: dict[str, Any], out_folder: str = shared_config.path_to_result
) -> None:
  """Saves the results to a JSON file."""
  out_name = f'{_DATE_AND_TIME}-correlation_vs_factscore.json'
  out_path = os.path.join(out_folder, out_name)
  utils.save_json(out_path, results)
  utils.print_info(f'Saved result to:\n{out_path}', add_punctuation=False)
  utils.print_divider()


def main(_) -> None:
  start_time = time.time()
  all_factscore_data = []

  for path in utils.listdir_wrapped(_FACTSCORE_DATA_FOLDER):
    if '_' not in path and path.endswith('.jsonl'):
      filepath = os.path.join(_FACTSCORE_DATA_FOLDER, path)
      all_factscore_data += load_factscore_data(filepath)

  all_factscore_data = utils.random_selection(
      in_list=all_factscore_data,
      num_to_select=_SAMPLES.value,
      random_seed=shared_config.random_seed,
  )
  utils.print_info(f'Evaluating on {len(all_factscore_data)} samples.')
  rater_model = modeling.Model(
      safe_config.model,
      temperature=safe_config.model_temp,
      max_tokens=safe_config.max_tokens,
  )
  all_results = {
      'eval_method': EVAL_METHOD,
      'samples': _SAMPLES.value,
      **utils.get_attributes(safe_config),
  }

  for do_step, step_name in [
      (_EVAL_IDENTIFY_ATOMIC_FACTS_ABLATED.value, IDENTIFY_FACTS),
      (_EVAL_RATE_ATOMIC_FACTS_ABLATED.value, RATE_FACTS),
      (_EVAL_ENTIRE_SAFE.value, ENTIRE_AUTORATER)
  ]:
    if not do_step:
      continue

    utils.print_divider()
    utils.print_info(f'Starting evaluation of `{step_name}`.')
    response_level_scores, per_prompt_data = run_eval_method(
        all_factscore_data=all_factscore_data,
        rater_model=rater_model,
        safe_step=step_name,
        eval_in_parallel=_EVAL_IN_PARALLEL.value,
    )
    utils.print_info(
        f'Completed evaluation of `{step_name}` in'
        f' {round(time.time() - start_time)} seconds.'
    )
    correlation_scores = compute_correlation(response_level_scores, step_name)
    print_correlation_results(correlation_scores=correlation_scores)
    all_results[step_name] = {
        **response_level_scores,
        'per_prompt_data': per_prompt_data,
        'correlation_scores': correlation_scores,
    }

  if _SAVE_RESULTS.value:
    save_results(all_results)


if __name__ == '__main__':
  app.run(main)
