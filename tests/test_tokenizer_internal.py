from cs336_basics.tokenizer import tokenize, pre_tokenize
import asyncio
import logging
import logging.config
import yaml
import sys

def test_example_in_pdf():
    tokens = {
        "low": 5,
        "lower": 2,
        "widest": 3,
        "newest": 6,
    }
    vocab, merges = tokenize(tokens, 6, ["<|special_token_1|>", "<|special_token_2|>"])
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

    vocab, merges = tokenize(tokens, 5)
    print(vocab)
    print(merges)


def test_word_count_simple():
    test_file = "/home/ying-zhang/workplace/cs336/cs336-assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
    result = pre_tokenize(file_name=test_file, parallelism=4, special_tokens=["<|endoftext|>", "<|test_special_token|>"])
    print(f"Got word count with {len(result)} entries")
    result_sorted = [(count, word) for word, count in result.items()]
    result_sorted.sort(reverse=True)
    print(result_sorted[:10])


def inspect_test_train_bpe_special_tokens():
    pkl_file = "/home/ying-zhang/workplace/cs336/cs336-assignment1-basics/tests/_snapshots/test_train_bpe_special_tokens.pkl"
    import pickle
    with open(pkl_file, 'rb') as f:
        loaded_data = pickle.load(f)
    print(loaded_data)


# Define the path to your configuration file
CONFIG_PATH = 'logging_config.yaml'

def setup_logging():
    """Load the logging configuration from the YAML file."""
    try:
        with open(CONFIG_PATH, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    except FileNotFoundError:
        print(f"Error: Logging configuration file not found at {CONFIG_PATH}", file=sys.stderr)
        # Fallback to a basic configuration if the file isn't found
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s')
    except Exception as e:
        print(f"Error loading logging configuration: {e}", file=sys.stderr)
        # Fallback in case of parsing errors
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s')

def main():
    # 1. Setup logging before any other operations
    setup_logging() 
    
    # 2. Get the root logger
    # Since we configured the root logger, calling getLogger() with no arguments returns it.
    logger = logging.getLogger() 
    
    # 3. Test the logger
    logger.info("This is an INFO message from the main function.") 
    logger.warning("A warning occurred here.")
    logger.debug("This message should NOT appear because the level is set to INFO.")

    test_example_in_pdf()
    test_word_count_simple()
    #inspect_test_train_bpe_special_tokens()


if __name__ == "__main__":
    main()