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
"""Shared configuration across SAFE files."""

from common import shared_config


################################################################################
#                               MODEL SETTINGS
# model_short: str = model for fact checking.
# model_temp: float = temperature to use for fact-checking model.
# max_tokens: int = max decode length.
# Currently supported models (copy-paste into `model_short` field): [
#     'gpt_4_turbo',
#     'gpt_4',
#     'gpt_4_32k',
#     'gpt_35_turbo',
#     'gpt_35_turbo_16k',
#     'claude_3_opus',
#     'claude_3_sonnet',
#     'claude_21',
#     'claude_20',
#     'claude_instant',
# ]
################################################################################
model_short = 'gpt_35_turbo'
model_temp = 0.1
max_tokens = 512

################################################################################
#                              SEARCH SETTINGS
# search_type: str = Google Search API used. Choose from ['serper'].
# num_searches: int = Number of results to show per search.
################################################################################
search_type = 'serper'
num_searches = 3

################################################################################
#                               SAFE SETTINGS
# max_steps: int = maximum number of break-down steps for factuality check.
# max_retries: int = maximum number of retries when fact checking fails.
# debug_safe: bool = show debugging printouts when running SAFE.
################################################################################
max_steps = 5
max_retries = 10
debug_safe = False

################################################################################
#                         FORCED SETTINGS, DO NOT EDIT
# model: str = overridden by full model name.
################################################################################
model = shared_config.model_options[model_short]
