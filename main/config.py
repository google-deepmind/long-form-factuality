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
"""Shared configuration across main experimentation files."""

from common import shared_config


################################################################################
#                             PIPELINE SETTINGS
# side_1: str = prompting method to use for side 1.
# side_2: str = prompting method to use for side 2.
# parallelize: bool = parallelize pipeline (turn on for faster bulk inference).
# save_results: bool = whether to save results to an output JSON.
#
# Supported prompting methods (copy-paste into `side` fields): [
#     'vanilla_prompting',
#     'naive_factuality_prompt',
#     'punt_if_unsure',  (punt and say "I don't know" if unsure)
#     'placeholder', (turns off a side and returns a placeholder response)
#     'none',  (turns off a side and returns an empty response)
# ]
################################################################################
side_1 = 'placeholder'
side_2 = 'vanilla_prompting'
parallelize = False
save_results = True

################################################################################
#                               MODEL SETTINGS
# responder_model_short: str = model for getting structures.
# Currently supported models (copy-paste into fields): [
#     'gpt_4_turbo',
#     'gpt_4',
#     'gpt_4_32k',
#     'gpt_35_turbo',
#     'gpt_35_turbo_16k',
#     'claude_3_opus',
#     'claude_3_sonnet',
#     'claude_3_haiku',
#     'claude_21',
#     'claude_20',
#     'claude_instant',
# ]
################################################################################
responder_model_short = 'gpt_35_turbo'

################################################################################
#                               DEBUG SETTINGS
# show_responder_prompts: bool = show prompts sent to responder if debugging.
# show_responder_responses: bool = show responses from responder if debugging.
################################################################################
show_responder_prompts = False
show_responder_responses = False

################################################################################
#                               DATA SETTINGS
# task_short: str = task to test model on. Supports a path to a results json.
# shuffle_data: bool = determines if prompts should be shuffled.
# max_num_examples: int = maximum number of examples to use. -1 for no cap.
# add_universal_postamble: bool = whether to add a universal postamble.
#
# Currently supported tasks (copy-paste into `task` field): [
#     'custom',  (uses default prompts in common/data_loader.py)
#     'longfact_concepts',
#     'longfact_objects',
# ]
# Old LongFact versions can be used by setting `task` to their directory.
################################################################################
task_short = 'longfact_objects'
shuffle_data = True
max_num_examples = 250
add_universal_postamble = True

################################################################################
#                             ABLATION SETTINGS
# use_length_ablation: bool = whether to add postamble ablating response length.
# num_sentences: int = how many sentences to limit the response to.
# response_length_postamble: str = formatting prompt for fixing response length.
################################################################################
use_length_ablation = False
num_sentences = 1
response_length_postamble = (
    f'Respond in exactly {num_sentences} sentences.' if num_sentences > 1
    else 'Respond in exactly 1 sentence.'
)


################################################################################
#                         FORCED SETTINGS, DO NOT EDIT
# responder_model: str = overridden by full model name.
# task: Tuple[str, str, str] = overriden by model name and data fields.
################################################################################
responder_model = shared_config.model_options[responder_model_short]
task = (
    shared_config.task_options[task_short]
    if task_short in shared_config.task_options
    else task_short
)
