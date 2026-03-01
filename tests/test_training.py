import numpy as np
import cs336_basics.train as train
import cs336_basics.tokenizer as tok
from .common import (
    TINY_STORIES_TRAIN,
    TINY_STORIES_VALID,
    TINY_STORIES_SAMPLE_1,
    TRAINED_BPE_TINY_STORIES_VALID_VOCAB,
    TRAINED_BPE_TINY_STORIES_VALID_MERGES,
    TRAINED_BPE_TINY_STORIES_TRAIN_VOCAB,
    TRAINED_BPE_TINY_STORIES_TRAIN_MERGES,
)
from .adapters import run_train_bpe, get_tokenizer
import pytest


@pytest.mark.skip(reason="should run only once building the encoded vocab")
def test_train_bpe_using_valid():
    train_bpe_and_assert(
        TINY_STORIES_VALID,
        TRAINED_BPE_TINY_STORIES_VALID_VOCAB,
        TRAINED_BPE_TINY_STORIES_VALID_MERGES,
    )
    train_bpe_and_assert(
        TINY_STORIES_TRAIN,
        TRAINED_BPE_TINY_STORIES_TRAIN_VOCAB,
        TRAINED_BPE_TINY_STORIES_TRAIN_MERGES,
    )


def train_bpe_and_assert(training_data, vocab_path, merges_path):
    vocab, merges = run_train_bpe(
        training_data, 10000, special_tokens=["<|endoftext|>"]
    )

    tokenizer = get_tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
    tokenizer.to_files(vocab_path, merges_path)

    tokenizer_decoded = tok.Tokenizer.from_files(
        vocab_path, merges_path, special_tokens=["<|endoftext|>"]
    )
    assert tokenizer_decoded.vocab == vocab
    assert tokenizer_decoded.merges == merges


def test_overfit_one_sample_with_reference_bpe():
    tokenizer = tok.Tokenizer.from_files(
        TRAINED_BPE_TINY_STORIES_TRAIN_VOCAB,
        TRAINED_BPE_TINY_STORIES_TRAIN_MERGES,
        special_tokens=["<|endoftext|>"],
    )

    tokens = []
    with open(TINY_STORIES_SAMPLE_1, "rb") as f:
        for token in tokenizer.encode_iterable(f):
            tokens.append(token)
    
    data = np.array(tokens)

    training_config = train.TrainingConfig(
        name="test_overfit_one_sample",
        # Training config
        total_iterations=5000,
        batch_size=1,
        # Model config
        context_length=256,
        n_layers=4,
        d_model=512,
        d_ff=1344,
        vocab_size=10000,
        num_heads=16,
        theta=10000,
        device="cuda:0",
        dtype="float32",
        # Optimizer config
        adamw_alpha=1e-3,
        adamw_beta1=0.9,
        adamw_beta2=0.999,
        adamw_eps=1e-8,
        adamw_lambda=0.01,
        adamw_device="cuda:0",
        adamw_dtype="float32",
        # Checkpointer config
        checkpoint_every_n=100000,  # disabled
        # Logging config
        log_every_n_steps=100,
    )
    training_config.test_single_sample_overfit = True

    train.training_loop(data, training_config)
