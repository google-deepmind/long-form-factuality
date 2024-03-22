# FActScore

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![arxiv](https://img.shields.io/badge/arXiv-2305.14251-b31b1b.svg)](https://arxiv.org/abs/2305.14251)
[![PyPI version factscore](https://badge.fury.io/py/factscore.svg)](https://pypi.python.org/pypi/factscore/)
[![Downloads](https://pepy.tech/badge/factscore)](https://pepy.tech/project/factscore)

**Forked from https://github.com/shmsw25/FActScore with edits for compatability with our project.**

This is the official release accompanying our EMNLP 2023 paper, [FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation](https://arxiv.org/abs/2305.14251). FActScore is available as a PIP package as well.

If you find FActScore useful, please cite:
```
@inproceedings{ factscore,
    title={ {FActScore}: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation },
    author={ Min, Sewon and Krishna, Kalpesh and Lyu, Xinxi and Lewis, Mike and Yih, Wen-tau and Koh, Pang Wei and Iyyer, Mohit and Zettlemoyer, Luke and Hajishirzi, Hannaneh },
    year={ 2023 },
    booktitle = { EMNLP },
    url={ https://arxiv.org/abs/2305.14251 }
}
```

## Announcement
* **11/04/2023**: The data we release includes human annotations of factual precision reported in Section 3 of [the paper](https://arxiv.org/abs/2305.14251). If you want to download these human annotated data *only*, without other data, you can download it directly from [this Google Drive link](https://drive.google.com/file/d/1enz1PxwxeMr4FRF9dtpCPXaZQCBejuVF/view?usp=sharing). We are also releasing FActScore results of 12 different LMs reported in Section 4.3 of the paper, in case you want to obtain them without running the code. Please refer to [here](#factscore-results-of-the-unlabeled-data).

## Install
<!-- ```
conda create -n fs-env python=3.9
conda activate fs-env
pip install -r requirements.txt
``` -->

Make a new Python 3.7+ environment using `virtualenv` or `conda`.

```bash
pip install --upgrade factscore
python -m spacy download en_core_web_sm
```

## Download the data

```bash
python -m factscore.download_data --llama_7B_HF_path "llama-7B"
```

This command does the following.
1. Download the knowledge source and example data.
2. Take the LLAMA 7B model and reconstruct Inst-LLAMA. This requires having access to HuggingFace weights of the LLAMA-7B model, which are added to the `--llama_7B_HF_path` flag. Follow [this guide](https://huggingface.co/docs/transformers/main/model_doc/llama) in order to obtain those weights. Skip the `--llama_7B_HF_path` if you would only like to use the ChatGPT version of FActScore.

**Optional flags**:
- `--data_dir`: directory to store the knowledge source and example data. `.cache/factscore` by default.
- `--model_dir`: directory to store Inst-LLAMA weights. `.cache/factscore` by default.

**Troubleshooting**:
- If you get a `ERROR 429: Too Many Requests` error while downloading the DB file, please download the DB from [this Google Drive link](https://drive.google.com/file/d/1mekls6OGOKLmt7gYtHs0WGf5oTamTNat/view?usp=sharing) and place it under `--data_dir` (`.cache/factscore` by default).
- If everything else fails, consider downloading the files manually from [this link](https://drive.google.com/drive/folders/1bLHGu_imkZVtX6O0mpZ-G0-4ofTLM1ZA?usp=share_link) and placing them in `--data_dir` and `--model_dir`, see [`factscore/download_data.py`](factscore/download_data.py) for more details.


## Running FActScore using a command line

We expect running FActScore costs about $1 of the API cost per 100 sentences. For instance, if you have 100 generations, each with 5 sentences on average, it costs $5 in total.

```bash
python -m factscore.factscorer --input_path {input_path} --model_name {estimator_name} --openai_key {openai_key}
```

- `--input_path` can be something like `data/unlabeled/InstructGPT.jsonl`. It should be a `.jsonl` format where each line contains `topic` (a topic entity that corresponds to the Wikipedia title) and `output` (a generation from the model).
- `--model_name`: `retrieval+ChatGPT` and `retrieval+llama+npm` (You can also use `retrieval+ChatGPT+npm` or `retrieval+llama` but we recommend the former two.)
- `--openai_key`: File containing OpenAI API Key.

**Optional flags**:
- `--data_dir`: Directory containing knowledge source, etc. `.cache/factscore` by default.
- `--model_dir`: Directory containing Inst-LLAMA weights. Skip if your `model_name` doesn't include `llama`. `.cache/factscore` by default.
- `--cache_dir`: Directory containing cache from API/models. `.cache/factscore` by default.
- `--use_atomic_facts`: If specified, it uses model-generated atomic facts released as part of our data instead of running the atomic fact generator. This will allow reproducing our results with no (or little if it still uses ChatGPT) cost. You can't specify it if you are running new model generations.
- `--gamma`: A hyperparameter for length penalty. `10` by default. It penalizes the score if the number of facts is less than `gamma`. `10` roughly corresponds to 2 sentences, so would penalize if the generation has less than 2 sentences. Usually, this would not change the ranking between systems unless some systems generate overly short responses all the time (e.g., models trained on NLP datasets without long-form generation tasks may do so). If you would like to turn off the length penalty completely, specify `--gamma 0`.
- `--n_samples`: If specified, it runs the model on a subset of the data.
- `--verbose`: If specified, it shows the progress bar.
- `--print_rate_limit_error`: It specified, it prints out rate limit errors from OpenAI API.
- `--cost_estimate`: This flag decides the type of OpenAI API cost estimation that we provide before calling it. It can be `"consider_cache"` (default) or `"ignore_cache"`.
- `--abstain_detection`: This flag optionally enables automatic detection of abstained responses. By default this is disabled, but it is recommended to add your own function tailored to your model. The currently supported detectors are `"generic"` and `"perplexity_ai"`, and their implementations can be found in [`factscore/abstain_detection.py`](factscore/abstain_detection.py). There are two methods to add your own abstain function: a) clone our GitHub repository to install `factscore` locally (`pip install --editable .`), and then add your function to [`factscore/abstain_detection.py`](factscore/abstain_detection.py) directly; b) process your abstain detection outside our package, and use empty strings in the `output` key for the JSONL file used in `--input_path`.
- `--knowledge_source`: In case the default knowledge source (Wikipedia - 2023/04/01) will not be used, preprocess it using the [instructions below](#To-use-a-custom-knowledge-source), and then specify the knowledge_source name under this flag.

## To evaluate your own LM

There're two sets of prompt entities, `data/labeled/prompt_entities.txt` (183 entities) and `data/unlabeled/prompt_entities.txt` (500 entities). Each line contains the name of the person (which is also a corresponding Wikipedia title). You can use the labeled version if you want to be compatible with the data under `data/labeled` (Section 3 and Section 4.2 in the paper), and use the unlabeled version if you want to be compatible with the data under `data/unlabeled` (Section 4.3 in the paper).

You can prompt your LM with your own prompt (we used `Question: Tell me a bio of <entity>.`) and use the following code.

```python
from factscore.factscorer import FactScorer

fs = FactScorer(openai_key="...")

# topics: list of strings (human entities used to generate bios)
# generations: list of strings (model generations)
out = fs.get_score(topics, generations, gamma=10)
print (out["score"]) # FActScore
print (out["init_score"]) # FActScore w/o length penalty
print (out["respond_ratio"]) # % of responding (not abstaining from answering)
print (out["num_facts_per_response"]) # average number of atomic facts per response
```

Alternatively, you can create a .jsonl file, where each line has `topic` (entity name, exactly same as the one from `.txt` file) and `output` (generation from LM), and then use a command line [above](#Running-FActScore-using-a-command-line).

We recommend using (A) `FactScorer(model_name="retrieval+ChatGPT")` (default) or (B) `FactScorer(model_name="retrieval+llama+npm")`. They have 0.99 Pearson correlation. Here're results of a range of models, which you can easily reproduce through [these command lines](#Running-FActScore-using-a-command-line).

| Model | % respond | # facts | FActScore from (A) | FActScore from (B) |
|---|---|---|---|---|
| [GPT-4](https://arxiv.org/abs/2303.08774)                                         | 88.2 | 60.8 | 73.1 | 59.9 |
| [ChatGPT](https://openai.com/blog/chatgpt)                                        | 84.2 | 37.0 | 71.6 | 60.4 |
| [Alpaca 65B](https://crfm.stanford.edu/2023/03/13/alpaca.html)                    | 100.0 | 17.1 | 55.6 | 46.3 |
| [InstructGPT](https://openai.com/research/instruction-following)                  | 99.8 | 27.7 | 52.8 | 41.7 |
| [Alpaca 13B](https://crfm.stanford.edu/2023/03/13/alpaca.html)                    | 100.0 | 16.6 | 47.7 | 40.3 |
| [Vicuna 13B](https://lmsys.org/blog/2023-03-30-vicuna/)                           | 76.6 | 50.9 | 46.6 | 40.7 |
| [Alpaca 7B](https://crfm.stanford.edu/2023/03/13/alpaca.html)                     | 100.0 | 17.4 | 39.7 | 36.5 |
| [Vicuna 7B](https://lmsys.org/blog/2023-03-30-vicuna/)                            | 91.0 | 45.6 | 38.9 | 36.9 |
| [MPT Chat 7B](https://www.mosaicml.com/blog/mpt-7b)                               | 88.8 | 37.3 | 30.1 | 27.9 |
| [Oasst Pythia 12B](https://huggingface.co/OpenAssistant/oasst-sft-1-pythia-12b)   | 100.0 | 39.7 | 25.1 | 20.8 |
| [Dolly 12B](https://huggingface.co/databricks/dolly-v2-12b)                       | 100.0 | 24.6 | 21.7 | 17.1 |
| [StableLM tuned 7B](https://huggingface.co/stabilityai/stablelm-tuned-alpha-7b)   | 66.6 | 38.0 | 17.3 | 16.3 |

`% respond` (% of responding instead of abstaining from answering) and `# facts` (# of atomic facts per valid response) indicate "factual recall" (how many pieces of information the model gives) and FActScore indicates "factual precision" (how accurate each piece of information the model gives is).

## To use a custom knowledge source

By default, FActScore uses Wikipedia dump from 2023/04/01. But you can also use your own knowledge source!

The knolwedge source should be ready in a `.jsonl` format, where each line is a dictionary containing `title` and `text`. `text` can either be a string or a list of strings (e.g., sections).

```python
from factscore.factscorer import FactScorer

fs = FactScorer()

# this will create a database using your file
# for English Wikipedia (18GB)), it takes ~8 hours
# once DB file is created, you can reuse it by only specifying `db_path`
fs.register_knowledge_source(name_of_your_knowledge_source,
                             data_path=path_to_jsonl_file,
                             db_path=path_to_output_db_file)

# now, when you compute a score, specify knowledge source to use
out = fs.get_score(topics, generations, knowledge_source=name_of_your_knowledge_source)
print (out["score"]) # FActScore
print (out["respond_ratio"]) # % of responding (not abstaining from answering)
print (out["num_facts_per_response"]) # average number of atomic facts per response
```

To see an example of constructing the ACL anthology knowledge source, see [`preprocessing/preprocess_acl.py`](preprocessing/preprocess_acl.py).

## FActScore results of the unlabeled data

You can easily reproduce FActScore results of 12 different LMs reported in Section 4.3 of [the paper](https://arxiv.org/abs/2305.14251) using this code. However, if you would like to obtain their predictions without running the code, you can download it from [this Google Drive link](https://drive.google.com/file/d/128qpNFhXJJTmPIbtqMJ5QSZprhWQDCDa/view?usp=sharing).

Each file corresponds to the subject LM (LM that generates responses that we are validating). Each line is a dictionary:
- `prompt`: the initial prompt fed into the LM
- `facts`: atomic facts decomposed by the model
- `LLAMA+NP_labels`: labels to facts, verified by LLAMA+NP
- `ChatGPT_labels`: labels to facts, verified by ChatGPT

Note that the number of lines may be less than 500, because it excludes the cases where the model abstains from responding (e.g., it says "I don't know"). You can do `# of lines / 500` to calculate the response ratio.

If you unzip the data and run the following code for verification, you will be able to get statistics that exactly match the statistics reported in the paper (Table 5 and Figure 3).
```python
dirname = "factscore-unlabeled-predictions"
for fn in os.listdir(dirname):
    chatgpt_fs = []
    llama_fs = []
    n_facts = []
    with open(os.path.join(dirname, fn)) as f:
        for line in f:
            dp = json.loads(line)
            n_facts.append(len(dp["facts"]))
            if "ChatGPT_Labels" in dp:
                chatgpt_fs.append(np.mean([l=="S" for l in dp["ChatGPT_Labels"]]))
            llama_fs.append(np.mean([l=="S" for l in dp["LLAMA+NP_Labels"]]))
    print ("Model=%s\t(%.1f%% responding, %.1f facts/response)\tFactScore=%.1f (ChatGPT)\t%.1f (LLAMA)" % (
        fn.split(".")[0], len(n_facts)*100/500, np.mean(n_facts), np.mean(chatgpt_fs)*100, np.mean(llama_fs)*100
    ))
```
