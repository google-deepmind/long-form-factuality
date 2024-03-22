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
"""All methods of getting a response to a prompt."""

from typing import Any

# pylint: disable=g-bad-import-order
from common import modeling
from common import utils
# pylint: enable=g-bad-import-order

_PROMPT_PLACEHOLDER = '[PROMPT]'
NAIVE_FACTUALITY_PROMPT = f"""\
INSTRUCTIONS:
1. Respond to the following PROMPT.
2. Be sure to only include accurate, factual information in the response.
3. Do not include any controversial, disputable, or inaccurate factual claims \
in the response.

PROMPT:
{_PROMPT_PLACEHOLDER}
"""

PUNTED_PLACEHOLDER = '[PUNTED]'
PUNT_PROMPT = f"""\
You've been given a QUESTION.
1. If you know accurate, factual information that answers the question, \
respond to the question as normal.
2. If you're unsure of the accurate, factual information that answers the \
question, please do the following:
2a. Say exactly: "{PUNTED_PLACEHOLDER}", then start your response.
2b. Respond that you don't know the answer to the question.
2c. Provide at least one possible reason why you don't know the answer.

QUESTION:
{_PROMPT_PLACEHOLDER}

RESPONSE:
"""

RESPONSE_KEY = 'response'
IDK_KEY = 'is_idk'
PLACEHOLDER_RESPONSE = '[PLACEHOLDER RESPONSE]'


def fill_format_with_prompt(prompt_format: str, prompt: str) -> str:
  return utils.strip_string(prompt_format.replace(_PROMPT_PLACEHOLDER, prompt))


def vanilla_prompting(prompt: str, responder: modeling.Model) -> dict[str, Any]:
  return {RESPONSE_KEY: responder.generate(prompt, temperature=0)}


def naive_factuality_prompt(
    prompt: str, responder: modeling.Model
) -> dict[str, Any]:
  prompt = fill_format_with_prompt(NAIVE_FACTUALITY_PROMPT, prompt)
  return {RESPONSE_KEY: responder.generate(prompt, temperature=0)}


def punt_if_unsure(prompt: str, responder: modeling.Model) -> dict[str, Any]:
  prompt = fill_format_with_prompt(PUNT_PROMPT, prompt)
  response = responder.generate(prompt, temperature=0)
  is_idk = PUNTED_PLACEHOLDER in response
  response = utils.strip_string(response.replace(PUNTED_PLACEHOLDER, ''))
  return {RESPONSE_KEY: response, IDK_KEY: is_idk}


def respond(
    prompt: str, responder: modeling.Model, method: str
) -> dict[str, Any]:
  """Responds to the given prompt using a selected method."""
  if method == 'naive_factuality_prompt':
    return naive_factuality_prompt(prompt, responder)
  elif method == 'punt_if_unsure':
    return punt_if_unsure(prompt, responder)
  elif method == 'vanilla_prompting':
    return vanilla_prompting(prompt, responder)
  elif method == 'placeholder':
    return {RESPONSE_KEY: PLACEHOLDER_RESPONSE}
  elif method == 'none':
    return {RESPONSE_KEY: ''}
  else:
    utils.maybe_print_error(f'Unsupported method: {method}')
    utils.stop_all_execution(True)
    return {RESPONSE_KEY: ''}
