# Long-Form Factuality in Large Language Models

This is the official code release accompanying our paper ["Long-form factuality in large language models"](https://arxiv.org/abs/2403.18802).
This repository contains:

1. **LongFact**: A prompt set of 2,280 fact-seeking prompts requiring long-form responses.
2. **Search-Augmented Factuality Evaluator (SAFE)**: Automatic evaluation of model responses in long-form factuality settings.
3. **F1@K**: Extending F1 score to long-form settings using recall from human-preferred length.
4. **Experimentation pipeline** for benchmarking OpenAI and Anthropic models using LongFact + SAFE.

## Installation

First, clone our GitHub repository.

```bash
git clone https://github.com/google-deepmind/long-form-factuality.git
```

Then navigate to the newly-created folder.
```bash
cd long-form-factuality
```

Next, create a new Python 3.10+ environment using `conda`.

```bash
conda create --name longfact python=3.10
```

Activate the newly-created environment.

```bash
conda activate longfact
```

All external package requirements are listed in `requirements.txt`.
To install all packages, and run the following command.

```bash
pip install -r requirements.txt
```

## Usage
### LongFact
The full prompt set for LongFact is available in the `longfact/` folder.
See the README in `longfact/` for more details about the dataset.

To run the data-generation pipeline that we used to generate LongFact, use the following command.
Refer to the README in `data_creation/` for additional details about the data-generation pipeline.

```bash
python -m data_creation.pipeline
```

### SAFE
Our full implementation of SAFE is located in `eval/safe/`.
See the README in `eval/safe/` for more information about how SAFE works.

To run the pipeline for evaluating SAFE against FActScore human annotations, use the following command.
Refer to the README in `eval/` for additional details about this experiment.

```bash
python -m eval.correlation_vs_factscore
```

### Benchmarking models
To benchmark OpenAI and Anthropic models, first add your API keys to `common/shared_config.py` (see README in `common/` for more information; be sure not to publish these keys).
To obtain model responses for a given prompt set, use the following command.
Refer to the README in `main/` for additional details about our main experimentation pipeline.

```bash
python -m main.pipeline
```

Next, to evaluate prompt-response pairs from our main experimentation pipeline using SAFE, use the following command, making sure to add the path to the `.json` file containing the prompt-response pairs to be evaluated to the `--result_path` argument.

```bash
python -m eval.run_eval \
    --result_path=
```

## Unit Tests

Each file in this directory has a corresponding unit test with the `_test` suffix (e.g., `file.py` would have `file_test.py` for unit tests).
Run commands for individual tests are shown in the unit test files.
To run all unit tests, use the following command.

```bash
python -m unittest discover -s ./ -p "*_test.py"
```

## Citing this work

If you find our code useful, please cite our [paper](https://arxiv.org/abs/2403.18802):

```bibtex
@misc{wei2024long,
  title={Long-form factuality in large language models},
  author={Wei, Jerry and Yang, Chengrun and Song, Xinying and Lu, Yifeng and Hu, Nathan and Huang, Jie and Tran, Dustin and Peng, Daiyi and Liu, Ruibo and Huang, Da and Du, Cosmo and Le, Quoc V.},
  year={2024},
  url={https://arxiv.org/abs/2403.18802},
}
```

## License and disclaimer

Copyright 2024 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
