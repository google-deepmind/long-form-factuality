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
"""Prompts the language model to generate prompts for LongFact."""

import copy

# pylint: disable=g-bad-import-order
from common import modeling
from common import utils
from data_creation import config as data_creation_config
from data_creation import examples
# pylint: enable=g-bad-import-order

_PREAMBLE_CONCEPTS = f"""\
Instructions:
1. Ask a general question about a concept.
2. IMPORTANT: DO NOT ask about an object (such as a person, place, or event,\
etc.).
3. Ask about the concept without asking about any particular aspect of the\
concept (e.g. history or significance).
4. The concept should be very specific and niche within the topic of \
{examples.PLACEHOLDERS.topic}.
5. The question should require a long-form response that includes several \
specific details such as numbers, names, dates, etc.
6. Follow the question styles in the provided examples.
7. Wrap the question in square brackets.
"""
_PREAMBLE_OBJECTS = f"""\
Instructions:
1. Ask a general question about a specific object (such as a person, place, \
event, act, company etc.).
2. The object should be very specific and niche within the topic of \
{examples.PLACEHOLDERS.topic}.
3. IMPORTANT: DO NOT ASK ABOUT A CONCEPT (such as doctrines, theories, ideas, \
methods, principles, etc.).
4. Do not ask about any particular aspect of the object (such as history, \
significance, impact, role, purpose, etc.).
5. Follow the question styles in the provided examples.
6. Wrap the question in square brackets.
"""
_PREAMBLE_OBJECTS_MORAL_DISPUTES = f"""\
Instructions:
1. Ask a general question about a specific object (such as a person, place, \
event, act, company etc.) within the topic of {examples.PLACEHOLDERS.topic}.
2. IMPORTANT: DO NOT ASK ABOUT A CONCEPT (such as doctrines, theories, ideas, \
methods, principles, debates, etc.).
3. Wrap the question in square brackets.

EXAMPLES:
[What is the Stanford Prison Experiment?]
[Explain the use of the Atomic Bomb on during World War II.]
[What is the Facebook's data privacy scandal in 2018?]
[Tell me about the Thalidomide drug scandal in the 1950s and 1960s.]
"""
FORMAT = f"""\
TOPIC:
{examples.PLACEHOLDERS.topic}

QUESTION:
[{examples.PLACEHOLDERS.prompt}]
"""

OBJECT_SUBTASK = 'objects'
CONCEPT_SUBTASK = 'concepts'
MORAL_DISPUTES_TOPIC = 'moral disputes'
ANY_TOPIC = 'ANY TOPIC'
ANY_TOPIC_SHORTCUT = 'anything'


def construct_prompt(
    topic: str,
    examples_list: list[examples.Example],
    subtask: str,  # must be CONCEPT_SUBTASK or OBJECT_SUBTASK
) -> str:
  """Constructs a single prompt for generating additional prompts."""
  topic = ANY_TOPIC if topic == ANY_TOPIC_SHORTCUT else topic.upper().strip()
  preamble = (
      _PREAMBLE_OBJECTS if subtask == OBJECT_SUBTASK else _PREAMBLE_CONCEPTS
  )

  if topic.lower() == MORAL_DISPUTES_TOPIC and subtask == OBJECT_SUBTASK:
    preamble = _PREAMBLE_OBJECTS_MORAL_DISPUTES

  preamble = preamble.replace(examples.PLACEHOLDERS.topic, topic)
  constructed_examples = examples.fill_format(FORMAT, examples_list)
  eval_example = FORMAT.replace(examples.PLACEHOLDERS.topic, topic)
  eval_example = eval_example.replace(examples.PLACEHOLDERS.prompt, '')[:-3]
  return utils.join_segments(preamble, constructed_examples, eval_example)


def generate_single_prompt(
    topic: str,
    model: modeling.Model,
    examples_list: list[examples.Example],
    subtask: str,  # must be CONCEPT_SUBTASK or OBJECT_SUBTASK
) -> str:
  """Generates a single prompt."""
  full_prompt = construct_prompt(topic, examples_list, subtask)
  response = model.generate(
      full_prompt, do_debug=data_creation_config.generate_data_debug
  )
  return utils.extract_first_square_brackets(response)


def run(
    topic: str,
    generator: modeling.Model,
    subtask: str,  # must be CONCEPT_SUBTASK or OBJECT_SUBTASK
    num_prompts: int = data_creation_config.num_prompts_to_generate,
) -> list[str]:
  """Use a language model to generate and return `num_prompts` prompts."""
  examples_list = (
      copy.deepcopy(examples.OBJECT_EXAMPLES) if subtask == OBJECT_SUBTASK
      else copy.deepcopy(examples.CONCEPT_EXAMPLES)
  )
  result = set()

  while len(result) < num_prompts:
    prompt = ''

    while not prompt:
      prompt = generate_single_prompt(topic, generator, examples_list, subtask)

    result.add(prompt)
    examples_list.append(examples.Example(prompt=prompt, topic=topic))
    utils.print_progress(f'Generating for {topic}', len(result), num_prompts)

  return list(result)
