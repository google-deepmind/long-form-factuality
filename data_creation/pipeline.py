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
r"""Pipeline for generating LongFact data.

Run command:
```
python -m data_creation.pipeline \
    --force_output_dir=<path> \
    --override=False
```
"""

import datetime
import os

from absl import app
from absl import flags
import langfun as lf

# pylint: disable=g-bad-import-order
from common import longfact
from common import modeling
from common import shared_config
from common import utils
from data_creation import config as data_creation_config
from data_creation import generate_data
# pylint: enable=g-bad-import-order

_FORCE_OUTPUT_DIR = flags.DEFINE_string(
    name='force_output_dir',
    default=None,
    help='Force saves results in this directory.',
    required=False,
)
_OVERRIDE = flags.DEFINE_bool(
    name='override', default=False, help='Override existing files'
)

_TASK_NAME = data_creation_config.subtask.replace('_', '-')
_LONGFACT_OBJECTS = 'longfact_objects'
_LONGFACT_CONCEPTS = 'longfact_concepts'
DATE = datetime.datetime.now().strftime('%m-%d-%Y')
DATE_AND_TIME = datetime.datetime.now().strftime('%m-%d-%Y-%H-%M-%S')


def find_output_name(topic: str = '', task_name: str = _TASK_NAME) -> str:
  """Calculates the output filename."""
  topic_str = topic.strip().replace(' ', '-')
  name = f'{task_name}.jsonl'

  if 'longfact' in task_name:
    name = name.replace('.jsonl', f'_{topic_str}.jsonl')
  else:
    name = name.replace('.jsonl', f'_{DATE_AND_TIME}.jsonl')

  return name


def find_output_folder(forced_dir: str, task_name: str = _TASK_NAME) -> str:
  """Calculates the output folder."""
  if forced_dir:
    return forced_dir
  else:
    return os.path.join(
        shared_config.path_to_data,
        f'{task_name}_{data_creation_config.generator_shorthand}_{DATE}/',
    )


def save_results(
    generated_prompts: list[str],
    out_dir: str,
    out_name: str,
    override: bool,
) -> str:
  """Saves results to an output file."""
  out_path = os.path.join(out_dir, out_name)
  utils.make_directory_wrapped(out_dir)

  if utils.file_exists_wrapped(out_path):
    utils.maybe_print_error(f'Dataset file already exists at {out_path}.')

    if not override:
      utils.stop_all_execution(True)

  items = [{longfact.PROMPT_KEY: prompt} for prompt in generated_prompts]
  utils.write_to_jsonlines(items, out_path)
  return out_path


def generate_prompts_for_topics(
    topics: list[str],
    generator: modeling.Model,
    out_folder: str,
    subtask: (
        str
    ),  # must be generate_data.CONCEPT_SUBTASK or generate_data.OBJECT_SUBTASK
    override_files: bool,
    num_prompts_to_generate: int = data_creation_config.num_prompts_to_generate,
    do_save_results: bool = data_creation_config.save_results,
) -> None:
  """Concurrently generate prompts for multiple topics."""
  def generate_single_topic(topic_index: int) -> list[str]:
    return generate_data.run(
        topic=topics[topic_index],
        generator=generator,
        subtask=subtask,
        num_prompts=num_prompts_to_generate,
    )

  topic_indices_to_generate, completed = set(range(len(topics))), set()

  while topic_indices_to_generate:
    for index, prompts, error in lf.concurrent_map(
        func=generate_single_topic,
        parallel_inputs=topic_indices_to_generate,
        max_workers=len(topic_indices_to_generate),
    ):
      if not prompts or error:
        utils.maybe_print_error(error)
      else:
        if do_save_results:
          out_name = find_output_name(topics[index])
          out_path = save_results(prompts, out_folder, out_name, override_files)

          if out_path:
            utils.print_info(f'Saved results to: {out_path}')

        topic_indices_to_generate.remove(index)
        completed.add(index)
        utils.print_progress('All topics progress', len(completed), len(topics))


def main(_) -> None:
  generator = modeling.Model(
      model_name=data_creation_config.generator,
      temperature=data_creation_config.generation_temp,
      max_tokens=128,
      show_responses=data_creation_config.show_generator_responses,
      show_prompts=data_creation_config.show_generator_prompts,
  )

  if data_creation_config.subtask == _LONGFACT_CONCEPTS:
    topics = longfact.list_topics()
    subtask = generate_data.CONCEPT_SUBTASK
  elif data_creation_config.subtask == _LONGFACT_OBJECTS:
    topics = longfact.list_topics()
    subtask = generate_data.OBJECT_SUBTASK
  else:
    raise ValueError(f'Unknown subtask: {data_creation_config.subtask}')

  generate_prompts_for_topics(
      topics=topics,
      generator=generator,
      out_folder=find_output_folder(_FORCE_OUTPUT_DIR.value),
      subtask=subtask,
      override_files=_OVERRIDE.value,
  )


if __name__ == '__main__':
  app.run(main)
