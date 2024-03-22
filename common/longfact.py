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
"""Store all details about the LongFact datasets."""

import dataclasses
import os

# pylint: disable=g-bad-import-order
from common import shared_config
from common import utils
# pylint: enable=g-bad-import-order


@dataclasses.dataclass()
class LongFactDataset:
  """Class for storing details for a single LongFact dataset."""
  topic: str
  path: str = ''


LONGFACT_CONCEPTS_FOLDER = os.path.join(
    shared_config.root_dir,
    'longfact/longfact-concepts_gpt4_01-10-2024_noduplicates/',
)
LONGFACT_OBJECTS_FOLDER = os.path.join(
    shared_config.root_dir,
    'longfact/longfact-objects_gpt4_01-12-2024_noduplicates/',
)
DATASETS = [
    LongFactDataset(topic='20th century events'),
    LongFactDataset(topic='accounting'),
    LongFactDataset(topic='architecture'),
    LongFactDataset(topic='astronomy'),
    LongFactDataset(topic='biology'),
    LongFactDataset(topic='business ethics'),
    LongFactDataset(topic='celebrities'),
    LongFactDataset(topic='chemistry'),
    LongFactDataset(topic='clinical knowledge'),
    LongFactDataset(topic='computer science'),
    LongFactDataset(topic='computer security'),
    LongFactDataset(topic='economics'),
    LongFactDataset(topic='electrical engineering'),
    LongFactDataset(topic='gaming'),
    LongFactDataset(topic='geography'),
    LongFactDataset(topic='global facts'),
    LongFactDataset(topic='history'),
    LongFactDataset(topic='immigration law'),
    LongFactDataset(topic='international law'),
    LongFactDataset(topic='jurisprudence'),
    LongFactDataset(topic='machine learning'),
    LongFactDataset(topic='management'),
    LongFactDataset(topic='marketing'),
    LongFactDataset(topic='mathematics'),
    LongFactDataset(topic='medicine'),
    LongFactDataset(topic='moral disputes'),
    LongFactDataset(topic='movies'),
    LongFactDataset(topic='music'),
    LongFactDataset(topic='philosophy'),
    LongFactDataset(topic='physics'),
    LongFactDataset(topic='prehistory'),
    LongFactDataset(topic='psychology'),
    LongFactDataset(topic='public relations'),
    LongFactDataset(topic='sociology'),
    LongFactDataset(topic='sports'),
    LongFactDataset(topic='US foreign policy'),
    LongFactDataset(topic='virology'),
    LongFactDataset(topic='world religions'),
]
PROMPT_KEY = 'prompt'


def list_topics() -> list[str]:
  return [dataset.topic for dataset in DATASETS]


def load_datasets(
    datasets: list[LongFactDataset], prompt_key: str = PROMPT_KEY
) -> list[str]:
  """Joins a list of longfact datasets into one dataset."""
  master_dataset, num_expected = [], 0

  for dataset in datasets:
    if dataset.path:
      loaded_dataset = utils.read_from_jsonlines(dataset.path)
      num_expected += len(loaded_dataset)

      for data in loaded_dataset:
        master_dataset.append({prompt_key: data[prompt_key]})

  assert len(master_dataset) == num_expected
  return [data[prompt_key] for data in master_dataset]


def load_datasets_from_folder(
    directory: str, prompt_key: str = PROMPT_KEY
) -> list[str]:
  """Loads all LongFacts datasets from a folder."""
  datasets = []

  for file in os.listdir(directory):
    topic = file.split('_')[-1].replace('.jsonl', '')
    datasets.append(
        LongFactDataset(topic=topic, path=os.path.join(directory, file))
    )

  return load_datasets(datasets, prompt_key=prompt_key)


def load_longfact_concepts() -> list[str]:
  return load_datasets_from_folder(LONGFACT_CONCEPTS_FOLDER)


def load_longfact_objects() -> list[str]:
  return load_datasets_from_folder(LONGFACT_OBJECTS_FOLDER)
