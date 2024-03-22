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
"""Shared configuration across files for data creation."""

from common import shared_config


################################################################################
#                               MODEL SETTINGS
# generator_model: str = the model for generating data.
# generation_temp: float = temperature to use when generating responses.
# Currently supported models (copy-paste into `generator_model` field): [
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
generator_model = 'gpt_4'
generation_temp = 1.0

################################################################################
#                          DATA GENERATION SETTINGS
# subtask: str = whether to generate about concepts or objects. Choose from [
#     'longfact_concepts',
#     'longfact_objects',
# ]
# num_prompts_to_generate: int = number of prompts to generate.
# max_in_context_examples: int = maximum number of in-context examples to use.
# save_results: bool = whether to save results.
################################################################################
subtask = 'longfact_concepts'
num_prompts_to_generate = 60
max_in_context_examples = 10
save_results = True

################################################################################
#                               DEBUG SETTINGS
# generate_data_debug: bool = shows debugging info during data generation.
# show_generator_prompts: bool = show prompts sent to generator.
# show_generator_responses: bool = show responses from generator.
################################################################################
generate_data_debug = False
show_generator_prompts = True
show_generator_responses = True

################################################################################
#                         FORCED SETTINGS, DO NOT EDIT
# generator_shorthand: str = saveable name of the generator model.
# generator: str = overridden by full model name.
################################################################################
generator_shorthand = shared_config.model_string[generator_model]
generator = shared_config.model_options[generator_model]
