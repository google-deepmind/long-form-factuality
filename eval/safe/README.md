# SAFE: Search-Augmented Factuality Evaluation

## Overview

Search-Augmented Factuality Evaluation (SAFE) is our proposed method of automatically evaluating long-form factuality generations using a language model connected to Google Search.
Given a prompt-response pair and a language model, the SAFE pipeline does the following:
1. Uses the language model to split the long-form response into individual facts (based off [the code from FActScore](https://github.com/shmsw25/FActScore/blob/main/factscore/atomic_facts.py), implemented in `get_atomic_facts.py`).
2. Uses the language model to revise each individual fact to be self-contained (implemented in `classify_relevance.py`).
3. For each individual fact, uses the language model determines whether the fact is relevant for answering the prompt, with respect to the entire long-form response (implemented in `classify_relevance.py`).
4. For each individual fact that was determined to be relevant for answering the prompt, uses the language model and Google Search calls to determine whether the atomic fact is supported by Google Search results (implemented in `rate_atomic_fact.py`).

The metrics returned for a prompt-response pair are thus **(a)** the number of atomic facts in the response, **(b)** the number of supported atomic facts in the response, **(c)** the number of irrelevant atomic facts in the response, and **(d)** the number of unsupported atomic facts in the response.

Experimentally, we compared the ratings returned from applying steps 2 to 4 of SAFE on a set of individual facts from [FActScore](https://arxiv.org/abs/2305.14251) to their respective crowdsource human annotations.
We found that SAFE has roughly **72% agreement with human raters** and **wins 76% of disagreement cases** (compared to human win rate of 19%) on a subset of 100 randomly-sampled disagreement examples.
By these metrics, we posit that SAFE achieves **superhuman-level rating** in long-form factuality.

## Usage

The SAFE pipeline is implemented in the `main` function in `search_augmented_factuality_eval.py`.
To call SAFE, you need to pass in a prompt-response pair and a `modeling.Model` language model.

**Serper API Key**

Our implementation of SAFE uses [Serper](https://serper.dev/) to get Google Search results for search queries.
To use SAFE with Serper, you'll need to create a Serper account following [these instructions](https://serper.dev/signup).
Next, you'll need to generate a Serper API key from [the API Key section](https://serper.dev/api-key).
Finally, add your Serper API key into the `common/shared_config.py` file in the `serper_api_key` variable.

**OpenAI API Key**

If you use an OpenAI language model with SAFE, you'll need to specify your OpenAI API key.
This can be done by adding your OpenAI API key into the `common/shared_config.py` file in the `openai_api_key` variable.

## Cost

For one prompt-response pair, we estimate that our implementation of SAFE requires roughly 100 Serper API credits (for ~100 Google Search calls) and $0.10 of OpenAI API credits when using `GPT-3.5-Turbo-0125`.
100 Serper API credits costs no more than $0.10 ($0.03 if Serper API credits are bought in bulk).
Refer to [Serper](https://serper.dev/) for the most up-to-date credit costs.

This means that running SAFE on a single prompt-response pair costs approximately $0.20.
However, for longer responses, running SAFE may cost up to $0.40.
For comparison, [FActScore](https://arxiv.org/pdf/2305.14251.pdf) required $4.00 for human annotation on a single prompt-response pair.

For example, if you want to run SAFE on a set of 100 prompt-response pairs, it would cost approximately $20.00 to $40.00.

## Custom SAFE Configuration
The current settings in the `config.py` file are the ones we used to generate our published version of SAFE.
However, you can update these settings as you wish to change the outputted data.

**Model Settings:**

- `model_short`: The model to use for SAFE. This field should be directly copy-pasted from the listed supported models shown in `config.py`.  Note that these model names are shortcuts to the full model names that are listed in `common/shared_config.py`. For example, `gpt_4` links to `GPT-4-0613`. `gpt_35_turbo` by default.
- `model_temp`: The temperature used for model decodes. A higher temperature should result in more-unique outputs, which could be useful if the model is getting stuck in retry loops. `0.5` by default.

**Search Settings:**

- `num_searches`: The number of search results to return every time Google Search is called. A higher number should result in more-comprehensive evaluation but may also require more credits. `3` by default.

**SAFE Settings:**

- `max_steps`: The maximum number of unique Google Search queries that the model can try. A higher number should result in more-comprehensive evaluation but requires more model calls and search credits. `5` by default.
- `max_retries`: The maximum number of times that to try a function that may throw an error as a result of the language model not returning a properly-formatted answer or the Google Search API not returning a result. `10` by default.
- `debug_safe`: Allows you to print out debugging information from the SAFE pipeline to the terminal. May clutter the screen. `False` by default.

**Custom Google Search**

We use the Serper API to get Google Search results for our model.
If you want to add your own Google Search API, you can do so by modifying the `call_search()` function in `rate_atomic_fact.py`.
You'll need to have a function that takes in the input search query and returns the Google Search results as a string.
Here's an example function header of what you would need to implement.

```python
def my_search_function(search_query: str) -> str:
  # Implement this function.
```

Then simply call this function in `rate_atomic_facts.call_search()`.

## Unit Tests

Each file in this directory has a corresponding unit test with the `_test` suffix (e.g., `file.py` would have `file_test.py` for unit tests). Run commands for individual tests are shown in the unit test files. To run all unit tests in this directory, use

```bash
python -m unittest discover -s eval/safe/ -p "*_test.py"
```
