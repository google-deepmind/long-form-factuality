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
"""Pipeline for running SxS response methods.

Run command:
```
python -m main.pipeline
```
"""

import datetime
import io
import os
import sys
import time
import types
from typing import Any, Optional

from absl import app
import langfun as lf

# pylint: disable=g-bad-import-order
from common import data_loader
from common import modeling
from common import shared_config
from common import utils
from main import config as main_config
from main import methods
# pylint: enable=g-bad-import-order

_HEADERS = (f'Side 1: {main_config.side_1}', f'Side 2: {main_config.side_2}')
_PROMPT = 'prompt'
_CORRECT_ANSWERS = 'correct_answers'
_INCORRECT_ANSWERS = 'incorrect_answers'
_SIDE1 = 'side1'
_SIDE2 = 'side2'
_SIDE1_RESPONSE = 'side1_response'
_SIDE2_RESPONSE = 'side2_response'
_PER_PROMPT_DATA = 'per_prompt_data'
_TOTAL_RUNTIME = 'total_runtime'

OUT_PATH = os.path.join(
    shared_config.path_to_result,
    datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.json',
)


def maybe_add_postamble(
    prompt: str,
    add_postamble: bool = main_config.add_universal_postamble,
    postamble_to_add: str = shared_config.prompt_postamble,
    delimiter: str = ' ',
) -> Optional[str]:
  """Tries to add a postamble, throwing an error if already added."""
  if add_postamble:
    postamble_to_add = utils.strip_string(postamble_to_add)

    if postamble_to_add in prompt:
      utils.maybe_print_error(f'Prompt already contains postamble: {prompt}')
      utils.stop_all_execution(True)
    else:
      return utils.strip_string(f'{prompt}{delimiter}{postamble_to_add}')
  else:
    return prompt


def print_config(model_name: str, model: modeling.Model) -> None:
  """Prints out the current configuration of experiments."""
  utils.print_info(f'{model_name} settings.')
  model.print_config()
  print()


def get_per_prompt_result(
    prompt: str,
    correct_answers: str | list[str],
    incorrect_answers: str | list[str],
    progress: str,
    responder: modeling.Model,
    side_1_method: str = main_config.side_1,
    side_2_method: str = main_config.side_2,
) -> dict[str, Any]:
  """Gets SxS comparison between two methods for a single prompt."""
  utils.print_divider()
  prompt = maybe_add_postamble(
      prompt=prompt,
      add_postamble=main_config.add_universal_postamble,
      postamble_to_add=shared_config.prompt_postamble,
  )
  prompt = maybe_add_postamble(
      prompt=prompt,
      add_postamble=main_config.use_length_ablation,
      postamble_to_add=main_config.response_length_postamble,
      delimiter='\n',
  )
  prompt_print_out = f'---PROMPT {progress}: {prompt}---'
  utils.print_color(prompt_print_out, color='blue')
  result = {
      _PROMPT: prompt,
      _CORRECT_ANSWERS: correct_answers,
      _INCORRECT_ANSWERS: incorrect_answers,
  }

  for method, side_str in [(side_1_method, _SIDE1), (side_2_method, _SIDE2)]:
    for k, value in methods.respond(prompt, responder, method).items():
      result[f'{side_str}_{k}'] = value

  utils.print_side_by_side(
      list1=[result[_SIDE1_RESPONSE]],
      list2=[result[_SIDE2_RESPONSE]],
      headers=_HEADERS,
  )

  return result


def save_results(
    results: list[dict[str, Any]],
    additional_info: Optional[dict[str, Any]] = None,
    module: types.ModuleType = main_config,
) -> None:
  """Saves results as a JSON."""
  output_dict = utils.get_attributes(module)
  output_dict[_PER_PROMPT_DATA] = results

  if additional_info:
    output_dict.update(additional_info)

  utils.make_directory_wrapped(OUT_PATH)
  utils.save_json(OUT_PATH, output_dict)


def get_results(
    data: data_loader.DataPackage,
    responder: modeling.Model,
    start_time: float,
    parallelize_across_prompts: bool = main_config.parallelize,
    save_results_every_step: bool = main_config.save_results,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
  """Get SxS comparison results between two methods."""
  def get_prompt_results_wrapped_for_parallelization(
      prompt_data_and_index: tuple[str, str | list[str], str | list[str], int]
  ) -> dict[str, Any]:
    prompt, correct_answers, incorrect_answers, _ = prompt_data_and_index
    return get_per_prompt_result(
        prompt, correct_answers, incorrect_answers, '', responder
    )

  results = []

  if parallelize_across_prompts:
    utils.print_info('Running with parallelization.')
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()
    prompts = [prompt for prompt, _, _ in data.iterate()]
    corrects = [correct_answer for _, correct_answer, _ in data.iterate()]
    incorrects = [incorrect_answer for _, _, incorrect_answer in data.iterate()]
    to_run, completed = set(range(len(prompts))), set()

    while to_run:
      for prompt_and_index, result, error in lf.concurrent_map(
          get_prompt_results_wrapped_for_parallelization,
          [(prompts[i], corrects[i], incorrects[i], i) for i in to_run],
          max_workers=len(to_run),
          show_progress=show_progress,
      ):
        if not result or error:
          utils.maybe_print_error(error)
        else:
          results.append(result)
          _, _, _, index = prompt_and_index
          completed.add(index)
          to_run.remove(index)
          utils.print_progress('Running prompts', len(completed), len(prompts))

    sys.stdout = original_stdout
  else:
    for i, (prompt, correct_ans, incorrect_ans) in enumerate(data.iterate()):
      results.append(
          get_per_prompt_result(
              prompt=prompt,
              correct_answers=correct_ans,
              incorrect_answers=incorrect_ans,
              progress=f'{i + 1}/{data.num_items()}',
              responder=responder,
          )
      )

      if save_results_every_step:
        time_log = {_TOTAL_RUNTIME: time.time() - start_time}
        save_results(results, additional_info=time_log)
        utils.print_info(
            f'Saved first {i + 1}/{data.num_items()} results to:\n{OUT_PATH}',
            add_punctuation=False,
        )
      else:
        utils.print_progress('Running prompts', len(results), data.num_items())

  return results


def load_data(
    filepath: str = shared_config.path_to_data,
    shuffle_data: bool = main_config.shuffle_data,
    random_seed: int = shared_config.random_seed,
    max_num_examples: int = main_config.max_num_examples,
    task: str = main_config.task,
) -> data_loader.DataPackage:
  """Loads data."""
  data = data_loader.DataPackage()
  data.load_and_prepare(
      filepath=filepath,
      shuffle_data=shuffle_data,
      random_seed=random_seed,
      max_num_examples=max_num_examples,
      task=task,
  )
  utils.print_info(f'Number of prompts: {data.num_items()}.')
  return data


def get_and_record_runtime(start_time: float) -> float:
  """Gets and records total runtime."""
  total_runtime = time.time() - start_time
  hours = int(total_runtime // 3600)
  minutes = int((total_runtime % 3600) // 60)
  seconds = round(total_runtime % 60)
  utils.print_divider()
  utils.print_info(f'Total runtime = {hours}:{minutes:02d}:{seconds:02d}.')
  return total_runtime


def main(_) -> None:
  utils.print_info(f'Saving results to:\n{OUT_PATH}', add_punctuation=False)
  start_time = time.time()

  responder = modeling.Model(
      model_name=main_config.responder_model,
      max_tokens=1024,
      show_responses=main_config.show_responder_responses,
      show_prompts=main_config.show_responder_prompts,
  )
  print_config('Responder', responder)

  utils.print_info(f'Task used: {main_config.task}')
  data = load_data()
  utils.print_info(f'Side 1: {main_config.side_1}')
  utils.print_info(f'Side 2: {main_config.side_2}')

  # Collect and print all results side-by-side
  results = get_results(data, responder, start_time)
  utils.print_side_by_side(
      [r[_SIDE1_RESPONSE] for r in results],
      [r[_SIDE2_RESPONSE] for r in results],
      headers=_HEADERS,
  )
  total_runtime = get_and_record_runtime(start_time)

  if main_config.save_results:
    save_results(results, additional_info={_TOTAL_RUNTIME: total_runtime})
    utils.print_info(
        f'Saved final results to:\n{OUT_PATH}', add_punctuation=False
    )


if __name__ == '__main__':
  app.run(main)
