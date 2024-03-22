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
"""Use a search-augmented LLM to evaluate factuality."""

import collections
import dataclasses
from typing import Any

# pylint: disable=g-bad-import-order
from common import modeling
from common import utils
from eval.safe import classify_relevance
from eval.safe import get_atomic_facts
from eval.safe import rate_atomic_fact
# pylint: enable=g-bad-import-order

IRRELEVANT_LABEL = 'Irrelevant'
SUPPORTED_LABEL = rate_atomic_fact.SUPPORTED_LABEL
NOT_SUPPORTED_LABEL = rate_atomic_fact.NOT_SUPPORTED_LABEL

_MAX_PIPELINE_RETRIES = 3


class CheckedStatement:
  """Class for storing checked statements."""

  def __init__(
      self,
      sentence: str,
      atomic_fact: str,
      self_contained_atomic_fact: str,
      relevance_data: dict[str, Any] | None = None,
      rate_data: rate_atomic_fact.FinalAnswer | None = None,
      annotation: str = '',
  ):
    self.sentence = sentence
    self.atomic_fact = atomic_fact
    self.self_contained_atomic_fact = self_contained_atomic_fact
    self.relevance_data = relevance_data
    self.rate_data = rate_data
    self.annotation = annotation
    self.data = {
        'sentence': self.sentence,
        'atomic_fact': self.atomic_fact,
        'self_contained_atomic_fact': self.self_contained_atomic_fact,
        'relevance_data': self.relevance_data if self.relevance_data else None,
        'rate_data': (
            dataclasses.asdict(self.rate_data) if self.rate_data else None
        ),
        'annotation': self.annotation,
    }


def count_labels(checked_statements: list[CheckedStatement]) -> dict[str, int]:
  """Extract scores from the checked statements for a single response."""
  result_dict = collections.defaultdict(int)

  # Ensure that these labels are in the dictionary
  for label in [SUPPORTED_LABEL, IRRELEVANT_LABEL, NOT_SUPPORTED_LABEL]:
    result_dict[label] = 0

  for statement in checked_statements:
    if not isinstance(statement, CheckedStatement) or not statement.annotation:
      continue

    if statement.annotation.lower() == SUPPORTED_LABEL.lower():
      result_dict[SUPPORTED_LABEL] += 1
    elif statement.annotation.lower() == IRRELEVANT_LABEL.lower():
      result_dict[IRRELEVANT_LABEL] += 1
    elif statement.annotation.lower() == NOT_SUPPORTED_LABEL.lower():
      result_dict[NOT_SUPPORTED_LABEL] += 1
    else:
      result_dict[statement.annotation] += 1
      utils.maybe_print_error(
          f'Unknown statement factuality type: {statement.annotation}'
      )

  return dict(result_dict)


def classify_relevance_and_rate_single(
    prompt: str,
    response: str,
    sentence: str,
    atomic_fact: str,
    rater: modeling.Model,
) -> tuple[CheckedStatement, dict[str, Any], dict[str, Any]]:
  """Classify relevance of and rate a single atomic fact."""
  is_relevant, self_contained_atomic_fact, revised_fact_dict = (
      classify_relevance.main(
          prompt, response, atomic_fact=atomic_fact, model=rater
      )
  )

  if not is_relevant:  # no need to rate further
    checked_statement = CheckedStatement(
        sentence=sentence,
        atomic_fact=atomic_fact,
        self_contained_atomic_fact=self_contained_atomic_fact,
        relevance_data=revised_fact_dict,
        annotation=IRRELEVANT_LABEL,
    )
    return checked_statement, revised_fact_dict, {}

  rate_data, past_steps_dict = rate_atomic_fact.check_atomic_fact(
      atomic_fact=self_contained_atomic_fact, rater=rater
  )

  if not isinstance(rate_data, rate_atomic_fact.FinalAnswer):
    raise ValueError('No rate data found for atomic fact.')

  checked_statement = CheckedStatement(
      sentence=sentence,
      atomic_fact=atomic_fact,
      self_contained_atomic_fact=self_contained_atomic_fact,
      relevance_data=revised_fact_dict,
      rate_data=rate_data,
      annotation=rate_data.answer,
  )

  return checked_statement, revised_fact_dict, past_steps_dict


def classify_relevance_and_rate(
    prompt: str,
    response: str,
    sentences_and_atomic_facts: list[dict[str, Any]],
    rater: modeling.Model,
) -> dict[str, Any]:
  """Classify relevance of and rate all given atomic facts."""
  checked_statements, revised_fact_dicts, past_steps_dicts = [], [], []

  for sentence_data in sentences_and_atomic_facts:
    sentence = sentence_data['sentence']
    assert 'atomic_facts' in sentence_data
    assert isinstance(sentence_data['atomic_facts'], list)

    for atomic_fact in sentence_data['atomic_facts']:
      checked_statement, num_fails = None, 0
      revised_fact_dict, past_steps_dict = {}, {}

      while checked_statement is None and num_fails < _MAX_PIPELINE_RETRIES:
        try:
          checked_statement, revised_fact_dict, past_steps_dict = (
              classify_relevance_and_rate_single(
                  prompt=prompt,
                  response=response,
                  sentence=sentence,
                  atomic_fact=atomic_fact,
                  rater=rater,
              )
          )
        except Exception as e:  # pylint: disable=broad-exception-caught
          utils.maybe_print_error(e)
          checked_statement, revised_fact_dict, past_steps_dict = None, {}, {}
          num_fails += 1

      if isinstance(checked_statement, CheckedStatement):
        checked_statements.append(checked_statement)
        revised_fact_dicts.append(revised_fact_dict)
        past_steps_dicts.append(past_steps_dict)

  return {
      'checked_statements': [item.data for item in checked_statements],
      'revised_fact_jsonified_all': revised_fact_dicts,
      'past_steps_jsonified_all': past_steps_dicts,
      **count_labels(checked_statements=checked_statements),
  }


def main(prompt: str, response: str, rater: modeling.Model) -> dict[str, Any]:
  atomic_facts = get_atomic_facts.main(response=response, model=rater)
  rating_result = classify_relevance_and_rate(
      prompt=prompt,
      response=response,
      sentences_and_atomic_facts=atomic_facts['all_atomic_facts'],
      rater=rater,
  )
  return {
      'prompt': prompt, 'response': response, **atomic_facts, **rating_result
  }
