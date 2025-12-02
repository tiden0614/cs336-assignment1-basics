from cs336_basics.tokenizer import learn_merges
from .adapters import run_train_bpe
import pytest


def test_example_in_pdf():
    tokens = {
        "low": 5,
        "lower": 2,
        "widest": 3,
        "newest": 6,
    }
    vocab, merges = learn_merges(tokens, 6, ["<|special_token_1|>", "<|special_token_2|>"])
    print(vocab)
    print(merges)


def test_tokenize_func_simple():
    tokens = {
        "Hello": 1,
        ",": 1,
        " how": 1,
        " are": 1,
        " you": 1,
        "?": 1,
        " regex": 2,
        " youth": 2,
        " fourth": 3,
        " hollow": 1,
    }

    a = bytes("He", 'utf-8')
    b = bytes("ll", 'utf-8')
    c = (a, b)
    d = (bytes("He", 'utf-8'), bytes('l', 'utf-8'))
    print(c)
    print(list(a))
    print(list(b))
    print(d)
    print(list(d))

    vocab, merges = learn_merges(tokens, 5)
    print(vocab)
    print(merges)


def test_word_count_simple():
    test_file = "/home/ying-zhang/workplace/cs336/cs336-assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    vocab, merge_history = run_train_bpe(test_file, 10000, special_tokens=["<|endoftext|>"])
    print(f"Got word count with {len(vocab)} entries")
    result_sorted = [(count, word) for word, count in vocab.items()]
    result_sorted.sort(reverse=True)
    print(result_sorted[:200])


@pytest.mark.skip(reason="single use")
def test_inspect_test_train_bpe_special_tokens():
    pkl_file = "/home/ying-zhang/workplace/cs336/cs336-assignment1-basics/tests/_snapshots/test_train_bpe_special_tokens.pkl"
    import pickle
    with open(pkl_file, 'rb') as f:
        loaded_data = pickle.load(f)
    print(loaded_data)
