from cs336_basics.tokenizer import tokenize
import logging
import logging.config
import yaml
import sys

def test_tokenize_func_simple():
    tokens = {
        "Hello": 1,
        ",": 1,
        " how": 1,
        " are": 1,
        " you": 1,
        "?": 1,
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

    result = tokenize(tokens, 3)
    print(result)


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

    test_tokenize_func_simple()

if __name__ == "__main__":
    main()