# Data Creation

## Overview
This folder contains the code for generating LongFact prompts. All logic related to data generation can be found in `examples.py` and `generate_data.py`. The pipeline for running the data-generation process is located in `pipeline.py`. Custom configuration variables are located in `config.py`.

## Data Generation Process
To create LongFact, we first manually selected a set of 38 topics (see the README in the `longfact` folder for the full list of topics) that provide broad coverage of long-form factuality. For a given topic, we generate prompts by instructing `GPT-4-0613` to come up with questions that require long-form responses and that are about specific and niche concepts (LongFact-Concepts) or objects (LongFact-Objects).

During each model call, we provide a set of in-context exemplars, which starts as a manually-selected set of in-context exemplars (shown in `examples.py`), but gradually get replaced with generated prompts as each model call generates a new prompt.

The selected in-context exemplars and the given topic are inputted into the prompt templates shown in `generate_data.py`. In-context exemplars for both LongFact-Concepts and LongFact-Objects use the same format (see the `FORMAT` variable at the top of `generate_data.py`):

```
TOPIC:
{examples.PLACEHOLDERS.topic}

QUESTION:
[{examples.PLACEHOLDERS.prompt}]
```

On the other hand, there are different preambles containing the instructions used to tell the model what the task is. These preambles can be found at the top of `generate_data.py`, specifically `PREAMBLE_CONCEPTS` for LongFact-Concepts and `PREAMBLE_OBJECTS` for LongFact-Objects. `PREAMBLE_OBJECTS_MORAL_DISPUTES` is a specific set of instructions needed to obtain prompts that ask about specific objects for the `moral_disputes` topic.

## Usage
To run the data-generation pipeline, use
```bash
python -m data_creation.pipeline
```

**(Optional)** You can also specify the following arguments:
- `--force_output_dir`: Directory to force-save results in. If specified, guarantees that all generated data will be in this directory. `None` by default.
- `--override`: If True, will allow the script to override existing files if it tries to save generated data into a file path that already exists. `False` by default.

Running this script should allow you to generate a similar version of LongFact (not exactly identical because of non-deterministic model outputs). The outputted generated data will be saved in JSONL filepaths that will be printed to the terminal. See the README in the `longfact` folder for the structure of these JSONL files.

**OpenAI API Key**

If you're using an OpenAI language model (our implementation by default uses `GPT-4-0613`), you'll need to specify your OpenAI API key. This can be done by adding your OpenAI API key into the `common/shared_config.py` file in the `openai_api_key` variable.

## Cost
We expect that generating LongFact costs about $5 of API cost per 100 generated prompts if you use `GPT-4-0613`. For example, if you have 38 topics and you want to generate 100 prompts per topic, it would cost approximately `3800 prompts * ($5 / 100 prompts) = $190`.

## Custom Data Generation
The current settings in the `config.py` file are the ones we used to generate our published version of LongFact. However, you can update these settings as you wish to change the outputted data.

**Model Settings:**

- `generator_model`: The model to use for generating data. This field should be directly copy-pasted from the listed supported models shown in `config.py`.  Note that these model names are shortcuts to the full model names that are listed in `common/shared_config.py`. For example, `gpt_4` links to `GPT-4-0613`. `gpt_4` by default.
- `generation_temp`: The temperature used for model decodes. A higher temperature should result in more-distinct generated prompts. `1.0` by default.

**Data Generation Settings**:

- `subtask`: Allows you to choose whether to generate questions about *concepts* within a topic (LongFact-Concepts) or *objects* within a topic (LongFact-Objects).
- `num_prompts_to_generate`: The number of prompts that should be generated for each topic. `60` by default.
- `max_in_context_exemplars`: The maximum number of in-context exemplars to give the model. Be sure to not to increase this number such that prompts are longer than the model's context length. `10` by default.
- `save_results`: Whether to try to save the generated prompts. Turning this off can be useful for preventing file clutter when debugging. `True` by default.

**Debug Settings**:
- `generate_data_debug`: the primary flag that allows you to see inputs and outputs to the model as the code runs. Turning this on may make the terminal hard to read because of the large prompts and responses, so this flag should only be set when debugging. `False` by default.
- `show_generator_prompts`: if `generate_data_debug` is on, then this flag controls whether prompts sent to the model will be printed out. Turn this flag off if the prompts are unimportant for your debugging case. `True` by default.
- `show_generator_responses`: if `generate_data_debug` is on, then this flag controls whether model responses will be printed out. Turn this flag off if the responses are unimportant for your debugging case. `True` by default.

**Custom Topics**

To generate prompts for topics other than the 38 topics in `LongFact` or singular topics, simply change the `DATASETS` variable located in `common/longfact.py`. For example, if you want to generate prompts for only the topic of "computers," you could set
```python
DATASETS = [LongFactDataset(topic='computers')]
```

As another example, if you wanted to generate prompts for the topics of "computers," "GPUs," and "CPUs," you could use
```python
DATASETS = [
  LongFactDataset(topic='computers'),
  LongFactDataset(topic='GPUs'),
  LongFactDataset(topic='CPUs'),
]
```

## Unit Tests

Each file in this directory has a corresponding unit test with the `_test` suffix (e.g., `file.py` would have `file_test.py` for unit tests). Run commands for individual tests are shown in the unit test files. To run all unit tests in this directory, use

```bash
python -m unittest discover -s data_creation/ -p "*_test.py"
```
