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
"""In-context examples for data creation."""

from data_creation import config as data_creation_config


class Example:
  def __init__(self, prompt: str, topic: str):
    self.prompt = prompt
    self.topic = topic.upper()


class Placeholder:
  def __init__(self) -> None:
    self.prompt = '[PROMPT]'
    self.topic = '[TOPIC]'

  def list_placeholders(self) -> list[str]:
    return [self.prompt, self.topic]


PLACEHOLDERS = Placeholder()
CONCEPT_EXAMPLES = [
    Example(
        topic='People',
        prompt=(
            'What was the relationship like between Thomas Edison and Nicola'
            ' Tesla?'
        ),
    ),
    Example(
        topic='People',
        prompt=(
            'How did the works of Yasunari Kawabata affect Japan and the'
            ' broader world?'
        ),
    ),
    Example(
        topic='Events',
        prompt='Explain what a black swan event is.',
    ),
    Example(
        topic='Events',
        prompt=(
            'What is the interplay between pandemics and societal'
            ' breakthroughs?'
        ),
    ),
    Example(
        topic='Companies',
        prompt=(
            'How has the East India Company affected how modern companies do'
            ' business?'
        ),
    ),
    Example(
        topic='Companies',
        prompt=(
            'What are the distinguishing characteristics of a benefit'
            ' corporation?'
        ),
    ),
    Example(
        topic='Cities',
        prompt='Explain the concept of a sponge city in urban planning.',
    ),
    Example(
        topic='Cities',
        prompt=(
            'How has the geography of the Amazon rainforest affected the city'
            ' of Iquitos?'
        ),
    ),
    Example(
        topic='Animals',
        prompt=(
            'How does the mimicry defense mechanism work in the animal kingdom?'
        ),
    ),
    Example(
        topic='Animals',
        prompt='In what ways are an ecosystem affected by salmon runs?',
    ),
]
OBJECT_EXAMPLES = [
    Example(
        topic='People',
        prompt='Who is Quoc V. Le?',
    ),
    Example(
        topic='People',
        prompt='Who is Xochitl Gomez?',
    ),
    Example(
        topic='Events',
        prompt='What happened in the first modern Olympics?',
    ),
    Example(
        topic='Events',
        prompt='What happened during the 5th Academy Awards?',
    ),
    Example(
        topic='Companies',
        prompt='Tell me about the company Acorns.',
    ),
    Example(
        topic='Companies',
        prompt='Tell me about the Carnegie Steel Company.',
    ),
    Example(
        topic='Cities',
        prompt='Give me an introduction of the city of Bucaramanga.',
    ),
    Example(
        topic='Cities',
        prompt='Give me an introduction of the city of Khartoum.',
    ),
    Example(
        topic='Animals',
        prompt='Tell me about the Antiguan racer snake.',
    ),
    Example(
        topic='Animals',
        prompt='Tell me about the South Philippine Dwarf Kingfisher',
    ),
]


def fill_format(prompt_format: str, examples: list[Example]) -> list[str]:
  """Fills a format with in-context examples."""
  if data_creation_config.max_in_context_examples <= 0:
    return []

  constructed_examples = []

  for example in examples[-data_creation_config.max_in_context_examples:]:
    curr_example = prompt_format.replace(PLACEHOLDERS.topic, example.topic)
    curr_example = curr_example.replace(PLACEHOLDERS.prompt, example.prompt)
    constructed_examples.append(curr_example)

  return constructed_examples
