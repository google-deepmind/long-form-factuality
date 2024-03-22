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
r"""Runs an eval pipeline on a result JSON from prompt-response generation.

The eval result of each prompt can be accessed in the way of
`dict_from_json[_PER_PROMPT_DATA][<prompt_index>]['side1_posthoc_eval_data']`.

Usage:
```
python -m eval.run_eval \
    --result_path=<path_to_result> \
    --eval_side1=True \
    --eval_side2=True \
    --parallelize=True \
    --max_claim=-1
```
"""

import copy
import datetime
import os
import threading
import time
from typing import Any

from absl import app
from absl import flags
import langfun as lf
import numpy as np

# pylint: disable=g-bad-import-order
from common import modeling
from common import shared_config
from common import utils
from eval import metric_utils
from eval.safe import config as safe_config
from eval.safe import search_augmented_factuality_eval as safe
# pylint: enable=g-bad-import-order

_RESULT_PATH = flags.DEFINE_string(
    'result_path', '', 'Path to the result file to eval.'
)
_EVAL_SIDE1 = flags.DEFINE_bool(
    'eval_side1', True, 'Whether to evaluate Side 1 response.'
)
_EVAL_SIDE2 = flags.DEFINE_bool(
    'eval_side2', True, 'Whether to evaluate Side 2 response.'
)
_PARALLELIZE = flags.DEFINE_bool(
    'parallelize', True, 'Whether to run eval in parallel.'
)
_MAX_CLAIM = flags.DEFINE_integer(
    'max_claim', -1, 'Maximum number of claims to consider when computing F1.'
)

_DO_NOT_RATE = ['none', 'placeholder']
_SIDE1 = 'side1'
_SIDE2 = 'side2'
_PER_PROMPT_DATA = 'per_prompt_data'
_PROMPT = 'prompt'
_SAVE_LOCK = threading.Lock()
_F1 = 'f1'

EVAL_KEY = 'posthoc_eval_data'
SIDE_TO_SIDE_STR = {_SIDE1: 'side_1', _SIDE2: 'side_2'}


def add_rating(
    prompt_result: dict[str, str | list[str] | dict[str, Any]],
    rater_model: modeling.Model,
    eval_side1: bool,
    eval_side2: bool,
) -> dict[str, str | list[str] | dict[str, Any]]:
  """Runs the eval pipeline on a single prompt's data."""
  prompt_result_rated = copy.deepcopy(prompt_result)

  for side, eval_side in [(_SIDE1, eval_side1), (_SIDE2, eval_side2)]:
    eval_side_key = f'{side}_{EVAL_KEY}'

    if eval_side and eval_side_key not in prompt_result_rated:
      prompt_result_rated[eval_side_key] = safe.main(
          prompt=prompt_result_rated[_PROMPT],
          response=prompt_result_rated[f'{side}_response'],
          rater=rater_model,
      )

  return prompt_result_rated


def evaluate_data(
    result_data: dict[str, Any],
    rater_model: modeling.Model,
    do_side1: bool,
    do_side2: bool,
    out_path: str,
    eval_in_parallel: bool,
    show_progress_bar: bool = True,
) -> None:
  """Runs the eval pipeline on a result JSON from prompt-response generation."""
  def add_rating_wrapped(
      single_prompt_result_and_index: tuple[dict[str, Any], int]
  ) -> dict[str, Any]:
    single_prompt_result = single_prompt_result_and_index[0]
    return add_rating(
        prompt_result=single_prompt_result,
        rater_model=rater_model,
        eval_side1=do_side1,
        eval_side2=do_side2,
    )

  if not do_side1 and not do_side2:
    return

  if eval_in_parallel:
    for prompt_data_and_index, result, error in lf.concurrent_map(
        add_rating_wrapped,
        [(item, i) for i, item in enumerate(result_data[_PER_PROMPT_DATA])],
        max_workers=25,
        show_progress=show_progress_bar,
    ):
      if error or not result:
        utils.maybe_print_error(error)
      else:
        _, i = prompt_data_and_index
        result_data[_PER_PROMPT_DATA][i] = result

        with _SAVE_LOCK:
          utils.save_json(out_path, result_data)
  else:
    for i, prompt_result in enumerate(result_data[_PER_PROMPT_DATA]):
      prompt_result_rated = add_rating_wrapped((prompt_result, i))
      result_data[_PER_PROMPT_DATA][i] = prompt_result_rated
      utils.save_json(out_path, result_data)
      utils.print_progress(
          'Evaluated prompt data', i + 1, len(result_data[_PER_PROMPT_DATA])
      )


def add_aggregation(
    per_prompt_data: list[dict[str, Any]], maybe_max_claims: int, eval_key: str
) -> None:
  for prompt_data in per_prompt_data:
    if (
        eval_key in prompt_data
        and safe.SUPPORTED_LABEL in prompt_data[eval_key]
        and safe.NOT_SUPPORTED_LABEL in prompt_data[eval_key]
        and safe.IRRELEVANT_LABEL in prompt_data[eval_key]
    ):
      f1_key = f'{_F1}_{maybe_max_claims}'
      prompt_data[eval_key][f1_key] = metric_utils.calculate_metrics(
          supported=prompt_data[eval_key][safe.SUPPORTED_LABEL],
          not_supported=prompt_data[eval_key][safe.NOT_SUPPORTED_LABEL],
          **({'max_claims': maybe_max_claims} if maybe_max_claims > 0 else {}),
      )


def print_results(
    result_data: dict[str, Any], maybe_max_claims: int
) -> None:
  """Prints the aggregated results of the eval pipeline."""
  for side in [_SIDE1, _SIDE2]:
    per_prompt_data = result_data[_PER_PROMPT_DATA]
    eval_side_key = f'{side}_{EVAL_KEY}'

    if eval_side_key not in per_prompt_data[0]:
      continue

    add_aggregation(per_prompt_data, maybe_max_claims, eval_side_key)
    utils.print_divider()
    utils.print_info(f'{result_data[SIDE_TO_SIDE_STR[side]]}')

    # Only compute averages for numerical evaluation metrics
    for metric, value in per_prompt_data[0][eval_side_key].items():
      if isinstance(value, float) or isinstance(value, int):
        all_values = [
            r[eval_side_key][metric] for r in per_prompt_data
            if eval_side_key in r
        ]
        avg_value, std_value = np.mean(all_values), np.std(all_values)
        result_data[f'{side}_avg_{metric}'] = avg_value
        result_data[f'{side}_std_{metric}'] = std_value
        utils.print_info(f'{metric.upper()}: AVG {avg_value}, STD {std_value}')


def main(_) -> None:
  start_time = time.time()
  utils.print_info(f'Running on {_RESULT_PATH.value}.')
  utils.print_info(f'SAFE search type: {safe_config.search_type}')
  utils.print_info(f'Evaluate `side 1` response: {_EVAL_SIDE1.value}')
  utils.print_info(f'Evaluate `side 2` response: {_EVAL_SIDE2.value}')
  out_folder = shared_config.path_to_result
  out_name = (
      datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '-SAFE.json'
  )
  out_path = os.path.join(out_folder, out_name)
  utils.print_info(f'Saving result to:\n{out_path}', add_punctuation=False)

  result_data = utils.read_json(_RESULT_PATH.value)
  rater_model = modeling.Model(
      safe_config.model,
      temperature=safe_config.model_temp,
      max_tokens=safe_config.max_tokens,
  )
  rater_model.print_config()
  result_data['autoeval_configs'] = utils.get_attributes(safe_config)
  do_side1 = _EVAL_SIDE1.value and result_data['side_1'] not in _DO_NOT_RATE
  do_side2 = _EVAL_SIDE2.value and result_data['side_2'] not in _DO_NOT_RATE
  utils.print_info(f'Evaluating {len(result_data[_PER_PROMPT_DATA])} prompts.')
  evaluate_data(
      result_data=result_data,
      rater_model=rater_model,
      do_side1=do_side1,
      do_side2=do_side2,
      out_path=out_path,
      eval_in_parallel=_PARALLELIZE.value,
  )
  utils.print_info(f'Finished in {round(time.time() - start_time)} seconds.')
  print_results(result_data, _MAX_CLAIM.value)
  utils.save_json(out_path, result_data)
  utils.print_divider()


if __name__ == '__main__':
  app.run(main)
