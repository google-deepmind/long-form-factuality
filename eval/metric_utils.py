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
"""Utility functions for metric calculation."""

import math

DEFAULT_MAX_CLAIMS = 100


def calculate_metrics(
    supported: int, not_supported: int, max_claims: int = DEFAULT_MAX_CLAIMS
) -> float:
  """Gets aggregated metrics."""
  if max_claims <= 0:
    raise ValueError('max_claims must be positive.')
  if supported < 0 or not_supported < 0:
    raise ValueError('Cannot have negative numbers of claims.')

  if supported == 0:
    return 0

  precision = float(supported / (supported + not_supported))
  recall = min(1, float(supported / max_claims))
  return 2 * precision * recall / (precision + recall)


def round_to_sigfigs(num_to_round: int | float, num_sigfigs: int) -> float:
  """Rounds num_to_round to the nearest number of significant figures."""
  if num_to_round == 0 or num_sigfigs <= 0:
    return 0
  elif math.isnan(num_to_round):
    return num_to_round
  else:
    return round(
        num_to_round,
        num_sigfigs - int(math.floor(math.log10(abs(num_to_round)))) - 1
    )
