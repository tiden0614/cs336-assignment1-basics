# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest

# print logs at INFO level and output to a file
clear && uv run pytest tests/test_train_bpe.py --log-cli-level=INFO -vv 2>&1 | tee test.log

clear && uv run pytest tests/test_tokenizer_internal.py --log-cli-level=INFO -vv
```

### Run UV commands and wait for VS code debugger

First, install debugpy
```sh
uv add --dev debugpy
```

Then, in VS Code, create a "Remote Attach" configuration in launch.json pointing 
to localhost:5678.

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Debugger: Attach",
            "type": "debugpy",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
            },
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}",
                    "remoteRoot": "."
                }
            ],
            "justMyCode": true
        }
    ]
}
```

Next, launch the whatever you want to run in terminal

```sh
uv run python -m debugpy --listen 5678 --wait-for-client -m pytest --log-cli-level=INFO -k test_overfit_one_sample_with_reference_bpe
```

Finally, attach VS code by running the remote debugger configured.

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

