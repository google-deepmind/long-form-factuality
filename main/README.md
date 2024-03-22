# Main Experimentation Pipeline

## Overview

This folder contains all code for getting model responses to prompts.

## Usage

To run the main experimentation pipeline, use

```bash
python -m main.pipeline
```

**Selecting prompting methods**

The experimentation pipeline is set up to allow side-by-side comparison between two methods and can also be configured to a single method by turning off a side.
There are two variables controlling each side: `side_1` and `side_2` in `config.py`.
To do a side-by-side comparison between two methods, set `side_1` to one method and `side_2` to the other method.
To only run on one method, set `side_1` or `side_2` to `'none'` or `'placeholder'` and then set the other side to the desired method.

Methods can be selected from any currently-supported method.
Support for additional methods can be added in `methods.py` following the existing methods as examples.
Currently-supported methods are:
- `'vanilla_prompting'`: prompt the model with the given prompt and obtain a response using a temperature of zero.
- `'naive_factuality_prompt'`: add instructions to only include accurate information in the response.
- `'punt_if_unsure'`: add instructions to punt if the model is unsure of the accurate information for responding to the prompt.
- `'placeholder'`: do not prompt the model for a response, but return the string `'[PLACEHOLDER RESPONSE]'` as the model response.
- `'none'`: do not prompt the model for a response and return the empty string as the model response.

**(Optional)** You can also specify the following arguments in `config.py`:
- `parallelize`: Whether to get responses in parallel. Parallelization may reduce runtime, but is harder to debug and is not necessary if running the pipeline on a small number of prompts. `False` by default.
- `save_results`: Whether to save the results from running the pipeline to a `.json` file. `True` by default.

**OpenAI API Key**

If you're using an OpenAI language model, you'll need to specify your OpenAI API key.
This can be done by adding your OpenAI API key into the `common/shared_config.py` file in the `openai_api_key` variable.

**Anthropic API Key**

If you're using an Anthropic language model, you'll need to specify your Anthropic API key.
This can be done by adding your Anthropic API key into the `common/shared_config.py` file in the `anthropic_api_key` variable.

## Custom Configuration

**Changing the language model**

The default configuration is set to use `GPT-3.5-Turbo` to respond to prompts.
To use any other model, simply copy-paste the desired model into the `responder_model_short` variable in `config.py`.
If a model that you would like to test on is not listed in the currently-supported models, see the README in the `common/` folder for information on how to add support for additional models.
Currently-supported models:
- `gpt_4_turbo`: links to `gpt-4-0125-preview`.
- `gpt_4`: links to `gpt-4-0613`.
- `gpt_4_32k`: links to `gpt-4-32k-0613`.
- `gpt_35_turbo`: links to `gpt-3.5-turbo-0125`.
- `gpt_35_turbo_16k`: links to `gpt-3.5-turbo-16k-0613`.
- `claude_3_opus`: links to `claude-3-opus-20240229`.
- `claude_3_sonnet`: links to `claude-3-sonnet-20240229`.
- `claude_21`: links to `claude-2.1`.
- `claude_20`: links to `claude-2.0`.
- `claude_instant`: links to `claude-instant-1.2`.

By default, models are set to respond to prompts at a temperature of zero in order to improve reproducibility.
However, this can be overridden by modifying the temperatures passed into each method in `methods.py`.

To show prompts being sent to the language model and responses to prompts from the language model in real time, set the `show_responder_prompts` and `show_responder_responses` variables, respectively, in `config.py` to `True`.

**Changing the task**

The default configuration is set to use LongFact-Objects as the prompt set, but this can be modified by simply copy-pasting the desired task into the `task_short` variable in `config.py`.
If a task that you would like to test on is not listed in the currently-supported tasks, see the README in the `common/` folder for information on how to add support for additional tasks.
Currently-supported tasks:
- `'custom'`: uses default sample prompts in `common/data_loader.py`.
- `'longfact_concepts'`: uses LongFact-Concepts, located in the `longfact/` folder.
- `'longfact_objects'`: uses LongFact-Objects, located in the `longfact/` folder.

You can also pass in a path to a result `JSON` file outputted by a previous experimental run using this pipeline.
By doing so, the prompts used in that previous experimental run can be replicated in the currently-configured run.

By default, a random subset of `250` prompts from the entire prompt set will be evaluated.
Randomization can be toggled using the `shuffle_data` variable in `config.py`, and the number of prompts to evaluate can be set using the `max_num_examples` variable in `config.py`.

Additionally, we add a universal postamble to all prompts that encourages the model to gives as many specific details as possible.
This universal postamble will not be added by setting the `add_universal_postamble` variable in `config.py` to `False`.
The postamble can also be changed by changing the `prompt_postamble` variable in `common/shared_config.py`.

**Prompting for desired response length**

By default, we do not specify the desired response length and let models respond following their default behavior.
This can be modified such that a postamble that asks the model to have a desired response length.
To set this behavior, first set `use_length_ablation = True` in `config.py`.
Then, use `num_sentences` in `config.py` to control the desired number of sentences that the response should be.
Finally, you can tune the postamble format using `response_length_postamble` in `config.py`.

## Unit Tests

Each file in this directory has a corresponding unit test with the `_test` suffix (e.g., `file.py` would have `file_test.py` for unit tests). Run commands for individual tests are shown in the unit test files. To run all unit tests in this directory, use

```bash
python -m unittest discover -s main/ -p "*_test.py"
```
