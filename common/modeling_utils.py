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
"""Utility functions for supporting modeling."""

import contextlib
from typing import Any, ContextManager, Optional

import langfun as lf

# pylint: disable=g-bad-import-order
from common import utils
# pylint: enable=g-bad-import-order


def add_format(prompt: str, model: Any, model_name: str) -> str:
  """Adds model-specific prompt formatting if necessary."""
  if model_name and model is not None:
    return utils.strip_string(prompt)
  else:
    return prompt


# pylint: disable=g-bare-generic
def get_lf_context(
    temp: Optional[float] = None, max_tokens: Optional[int] = None
) -> ContextManager:
  """Gets a LangFun context manager with the given settings."""
  # pylint: enable=g-bare-generic
  @contextlib.contextmanager
  def dummy_context_manager():
    yield None

  if temp is not None and max_tokens is not None:
    return lf.use_settings(temperature=temp, max_tokens=max_tokens)
  elif temp is not None:
    return lf.use_settings(temperature=temp)
  elif max_tokens is not None:
    return lf.use_settings(max_tokens=max_tokens)
  else:
    return dummy_context_manager()
