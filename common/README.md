# Common Code Files

## Overview

The files in this folder contain shared code used across the codebase, including functions for:
- Loading data (implemented in `data_loader.py` and `longfact.py`).
- Unifying language models (implemented in `modeling.py` and `modeling_utils.py`).
- Shared configuration settings (implemented in `shared_config.py`).
- Additional utility functions (implemented in `utils.py`).

## Usage

This folder does not contain any code that should be directly run.

## Important Settings

Running other code files in this codebase will likely require an API key of some sort.
For example, running any code that uses OpenAI models will require you to pass your OpenAI API key.
Thus, any API key that you plan to use should be set in `shared_config.py`.
In particular, the `openai_api_key`, `anthropic_api_key`, and `serper_api_key` variables should store the API keys for [OpenAI](openai.com/), [Anthropic](console.anthropic.com), and [Serper](serper.dev), respectively.

## Custom Configuration

**Changing the random seed**

Whenever code in our codebase requires a `random` operation (e.g., `random.shuffle` or `random.select`), we pass a random seed to improve reproducibility.
By default, we use `1` as our random seed, but this can be modified as desired by setting the `random_seed` variable in `shared_config.py`.

**Changing the results folder**

Whenever code in our codebase requires saving data that resulted from running an experiment, we pass a root folder that will contain all result files.
By default, all result files will be saved in `results/`.
However, this can be modified by changing the `path_to_result` variable in `shared_config.py`.

**Adding a universal postamble**

When running models using our experimentation pipeline in the `main/` folder, you can optionally add a universal postamble to all prompts.
This allows you to, for example, prompt models to give more specific details when responding to factual queries.
By default, we include a postamble that prompts the model to give specific details and examples such as names, numbers, events, etc.
This postamble can be modified by setting the `prompt_postamble` variable in `shared_config.py`.
**Reminder**: This postamble will not be used if the option to use it is not set to `True` in `main/config.py`.

**Supporting additional models**

We natively provide support for the following models:
- GPT-4-Turbo (`gpt-4-0125-preview`)
- GPT-4 (`gpt-4-0613`)
- GPT-4-32K (`gpt-4-32k-0613`)
- GPT-3.5-Turbo (`gpt-3.5-turbo-0125`)
- GPT-3.5-Turbo-16K (`gpt-3.5-turbo-16k-0613`)
- Claude-3-Opus (`claude-3-opus-20240229`)
- Claude-3-Sonnet (`claude-3-sonnet-20240229`)
- Claude-3-Haiku (`claude-3-haiku-20240307`)
- Claude-2.1 (`claude-2.1`)
- Claude-2.0 (`claude-2.0`)
- Claude-Instant (`claude-instant-1.2`)

To add additional OpenAI models, simply add an entry of the desired shorthand name and the full model name with the prefix `'OPENAI:'` in the `model_options` variable in `shared_config.py` and an entry of the desired shorthand name and the saveable model name in the `model_string` variable in `shared_config.py`.
Likewise, to add additional Anthropic models, simply do the same as with an OpenAI model except with the prefix `'ANTHROPIC:'` instead.

The simplest way to add other models is to create a new class that inherits from the `langfun.LanguageModel` class, following the `AnthropicModel` class that is shown in `modeling.py`.
Then, add the new class into the `load()` method in the `Model` class in `modeling.py`.
Finally, connect the new class to the main experimentation pipeline by adding the relevant entries in the `model_options` and `model_string` variables in `shared_config.py`.

Alternatively, a new model class can be added without inheriting from the `langfun.LanguageModel` class by changing the `Model.generate()` function to include generation methods specific to this new class.
The new class should still be added into the `Model.load()` function and linked in `shared_config.py`.

**Supporting additional tasks**

We natively provide support to run on LongFact.

To run on other datasets, format the datasets as a `.jsonl` where each line stores a single prompt's data.
Each prompt's data should be a dictionary with the same set of keys storing information such as the prompt itself, correct answers (if desired), and incorrect answers (if desired).
Next, save this formatted dataset in the data folder that is specified by the `path_to_data` variable in `shared_config.py` (`datasets/` by default).
Finally, add an entry to the `task_options` variable in `shared_config.py` where the key is the shorthand name of the dataset and the value is a tuple containing (1) the filename (without `.jsonl`) of the dataset, (2) the dictionary key that stores each prompt, (3) the dictionary key that stores correct answers or `'NONE'` if there are no correct answers, and (4) the dictionary key that stores incorrect answers or `'NONE'` if there are no incorrect answers.
This dataset can then be accessed in the main experimentational pipeline by using the shorthand name that was just added.

As an example, if the following data was saved in `datasets/test_task.jsonl`, the `path_to_data` should remain as `datasets/`, and the added entry to the `task_options` variable would be `'test': ('test_task', 'prompt', 'Correct Answers', 'NONE')`.
After these changes, the data can be loaded by setting `task_short = 'test'` in `main/config.py`.

```json
{"prompt": "Here is a question.", "Correct Answers": "Here is a correct answer."}
{"prompt": "Here is another question.", "Correct Answers": "Here is the corresponding correct answer."}
```

## Unit Tests

Each file in this directory has a corresponding unit test with the `_test` suffix (e.g., `file.py` would have `file_test.py` for unit tests). Run commands for individual tests are shown in the unit test files. To run all unit tests in this directory, use

```bash
python -m unittest discover -s common/ -p "*_test.py"
```
