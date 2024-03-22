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
"""Splits a response into atomic facts."""

import itertools
from typing import Any

# pylint: disable=g-bad-import-order
from common import modeling
from third_party.factscore import atomic_facts
# pylint: enable=g-bad-import-order

_SENTENCE = 'sentence'
_ATOMIC_FACTS = 'atomic_facts'


def convert_atomic_facts_to_dicts(
    outputted_facts: list[tuple[str, list[str]]]
) -> list[dict[str, Any]]:
  return [
      {_SENTENCE: sentence, _ATOMIC_FACTS: identified_atomic_facts}
      for sentence, identified_atomic_facts in outputted_facts
  ]


def main(response: str, model: modeling.Model) -> dict[str, Any]:
  atomic_fact_generator = atomic_facts.AtomicFactGenerator(
      api_key='', gpt3_cache_file='', other_lm=model
  )
  facts, _ = atomic_fact_generator.run(response)
  facts_as_dict = convert_atomic_facts_to_dicts(facts)
  all_atomic_facts = list(
      itertools.chain.from_iterable([f[_ATOMIC_FACTS] for f in facts_as_dict])
  )
  return {
      'num_claims': len(all_atomic_facts),
      'sentences_and_atomic_facts': facts,
      'all_atomic_facts': facts_as_dict,
  }
