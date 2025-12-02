import concurrent.futures
import logging
import os
import sys
import regex as re
import asyncio
from collections import defaultdict
from typing import BinaryIO

log = logging.getLogger("tokenizer")
TOKEN_SPLIT_PAT = br"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
END_OF_TEXT_TOK = "<|endoftext|>"
END_OF_TEXT_TOK_BYTES = END_OF_TEXT_TOK.encode('utf-8')

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


# Given a portion of the file, tokenize it using a regex
# and then put words into a counting dictionary.
#
# The overall algorithm is optimized towards keeping a limited working set
# in-memory instead of loading the whole chunk at once. This way, we can
# have many threads working concurrently without the heap memory exploding.
def word_count(file_name: str, start: int, end: int, special_token_pattern: bytes) -> dict[str, int]:
    special_token_pattern_str = special_token_pattern.decode('utf-8')

    LOAD_SIZE = 16 << 10 # 16KB
    with open(file_name, "rb") as file:
        file.seek(start)

        # Algorithm:
        # 1. We load LOAD_SIZE into memory and append to working_set 
        #    (list of strings). We continue loading until the latest
        #    mini_chunk contains a <|endoftext|> token.
        # 2. Split the lastest mini_chunk by the last <|endoftext|>,
        #    and append only the first part into the working_set.
        # 3. Now the working_set contains N complete paragraphs, we
        #    can start word counting on it.
        # 4. Repeat 1 (remember to get the 2nd split from step 2),
        #    until the file portion is exhausted.

        loaded_size = 0
        working_set: list[str] = [] # list of mini_chunks loaded so far
        pre_split = None
        chunks_loaded = 0
        word_count_dict = defaultdict(int)
        total_load_size = end - start
        while loaded_size <= total_load_size:
            load_size = min(LOAD_SIZE, total_load_size - loaded_size)
            log.debug(f"Loading {load_size >> 10}KB into memory")
            mini_chunk = file.read(load_size)
            chunks_loaded += 1
            loaded_size += load_size

            # If there were a previous split, prepend it into
            # the mini_chunk.
            if pre_split is not None:
                mini_chunk = pre_split + mini_chunk
                pre_split = None

            if len(mini_chunk) == 0:
                log.info(f"Encountered EOF after loading {loaded_size}. Collected " \
                         f"{len(word_count_dict)} word-freq pairs")

                special_token_start, special_token = 0, b"<|EOF|>"
            else:
                found_at = re.search(special_token_pattern, mini_chunk)
                if not found_at:
                    working_set.append(mini_chunk)
                    continue

                special_token_start, special_token = found_at.start(), found_at.group(0)

            # We have found a END_OF_TEXT_TOK from the loaded
            # mini_chunk, now split it into 2
            idx_of_first_char_after_tok = special_token_start + len(special_token)
            working_set.append(mini_chunk[:special_token_start])
            assert pre_split is None, f"Unprocessed {pre_split}"
            pre_split = mini_chunk[idx_of_first_char_after_tok:]
            log.debug("Found mini_chunk containing special token. "\
                     "chunks_loaded=%d mini_chunk=%d found_at=%d special_token=%s pre_split_len=%d",
                     chunks_loaded, len(mini_chunk), special_token_start, special_token, len(pre_split))
            
            to_word_count: bytes = b"".join(working_set)
            working_set.clear()

            log.debug(f"Starting to count words in to_word_count of size {len(to_word_count)}")
            scanner = re.finditer(TOKEN_SPLIT_PAT, to_word_count)
            for token in scanner:
                word_count_dict[token.group(0)] += 1
            
            if len(mini_chunk) == 0:
                break
        
        log.info(f"Finished portion. loaded_size={loaded_size} chunks_loaded={chunks_loaded} "\
                 f"tokens_count={len(word_count_dict)}")
        return word_count_dict


def validate_split_boundaries(parallelism: int, boundaries: list[int])-> list[tuple[int, int]]:
    assert len(boundaries) > 0
    assert len(boundaries) == parallelism + 1, (len(boundaries), parallelism)

    result: list[tuple[int, int]] = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        assert start < end, f"Found invalid start end pair {start},{end} at {i} for {boundaries}"
        result.append((start, end))
    
    assert parallelism == len(result)
    return result


# Glue code to convert a concurrent.futures.Future object 
# into a coroutine so that we can use the nice async programming
# constructs
async def _await_concurrent_future(fut: concurrent.futures.Future):
    return await asyncio.wrap_future(fut)


async def drive_concurrent_word_count(
        boundary_pairs: list[tuple[int, int]], 
        file_name: str, 
        parallelism: int,
        special_token_pattern: str) -> list[dict[str, int]]:
    with concurrent.futures.ProcessPoolExecutor(max_workers=parallelism) as executor:
        loop = asyncio.get_running_loop()
        tasks = []
        async with asyncio.TaskGroup() as tg:
            for start, end in boundary_pairs:
                fut = loop.run_in_executor(
                    executor, word_count, file_name, start, end, special_token_pattern)
                task = tg.create_task(_await_concurrent_future(fut))
                tasks.append(task)
    
    return [task.result() for task in tasks]

def pre_tokenize(file_name: str, parallelism: int, special_tokens: list[str]) -> dict[str, int]:
    # 0. build special tokens regex
    # 1. find boundaries
    # 2. for each boundary pair, assign a thread to do word count
    # 3. wait for all threads to finish, and then merge all outputs

    # Step 0: build special tokens regex
    log.info(f"Received special tokens {special_tokens}")
    special_tokens_pattern = "|".join(map(re.escape, special_tokens)).encode('utf-8')
    log.info(f"Using special tokens pattern {special_tokens_pattern}")

    # Step 1: find boundaries
    log.info(f"Splitting file {file_name} into {parallelism} chunks")
    with open(file_name, "rb") as f:
        boundaries = find_chunk_boundaries(f, parallelism, b"<|endoftext|>")
    
    # Double check the correctness of the split output
    boundary_pairs = validate_split_boundaries(parallelism, boundaries)

    log.info(f"Finished boundary probe. Got {len(boundary_pairs)} chunks.")

    # Step 2: spin up async word count workers
    try:
        results = asyncio.run(drive_concurrent_word_count(
            boundary_pairs, file_name, parallelism, special_tokens_pattern))
    except Exception as e:
        log.error(f"Encountered exception running async word count", e)
        sys.exit(1)
    log.info("All async word count threads have finished")
    
    # Step 3: merge into one word count dict
    merged_word_count: dict[str, int] = defaultdict(int)
    for word_count_dict in results:
        for word, count in word_count_dict.items():
            merged_word_count[word] += count

    return merged_word_count


def initialize_merged_tokens(tokens: dict[bytes, int]) -> dict[tuple[bytes], int]:
    merged_tokens = {}
    for token, count in tokens.items():
        if token == "":
            pass

        token_bytes_list = [c.to_bytes(1, 'big') for c in token]
        merged_tokens[tuple(token_bytes_list)] = count
        log.debug("token %s token_bytes_list %s", token, token_bytes_list)
    
    return merged_tokens
    

def initialize_current_encoding(special_tokens: list[str]) -> dict[int, bytes]:
    current_encoding = {}
    for i in range(len(special_tokens)):
        current_encoding[i] = special_tokens[i].encode('utf-8')
    for i in range(256):
        current_encoding[i + len(special_tokens)] = i.to_bytes(1, 'big')
    return current_encoding


def tokenize(
        tokens: dict[bytes, int], 
        passes: int, 
        special_tokens: list[str]) -> tuple[dict[bytes, int], list[tuple[bytes, bytes]]]:
    log.debug("Received tokens %s", tokens)
    # merged_tokens stores the partially compressed
    # tokens corresonding to their count
    # e.g.
    # {
    #   (l, o, w): 5,
    #   (w, i, d, e, st): 10,
    #   (l, o, w, e, st): 15,
    # }
    merged_tokens: dict[tuple[bytes], int] = initialize_merged_tokens(tokens)
    # log.info(f"Initial merged_tokens {merged_tokens}")

    # current_encoding stores the currently accepted
    # encodings.
    # it would look like something:
    # {
    #   l: 30,
    #   o: 31,
    #   ...
    #   st: 233,
    # }
    current_encoding: dict[int, bytes] = initialize_current_encoding(special_tokens)
    next_encoding = len(current_encoding)
    log.debug(f"Initial current_encoding {current_encoding}")

    merge_history: list[tuple[bytes, bytes]] = []

    for i in range(passes):
        log.debug(f"Performing pass {i}. current_encoding_size {len(current_encoding)}")

        # candidate_encoding stores new encodings (that don't exist
        # in current_encoding) associate with the count they appear
        candidate_encoding: dict[tuple[bytes, bytes], int] = defaultdict(int)

        for bytes_tuple, count in merged_tokens.items():
            assert len(bytes_tuple) > 0, "empty byte tuple"

            if len(bytes_tuple) == 1:
                # when the whole bytes_tuple contains only one element, it must
                # have already been merged into the current_encoding
                continue

            for i in range(len(bytes_tuple) - 1, 0, -1):
                candidate_encoding[(bytes_tuple[i - 1], bytes_tuple[i])] += count
        
        # now we have collected all candidate encodings, sort them based on
        # the count and get the most frequent sequence
        log.debug(f"Collected {len(candidate_encoding)} new candidates.")
        if len(candidate_encoding) == 0:
            log.info(f"No more candidates to merge.")
            break

        # Pick the most frequent bytes pair as merge candidate
        merge_candidate = (b"", b"")
        merge_candidate_count = 0
        for encoding_pair_candidate, current_count in candidate_encoding.items():
            if current_count > merge_candidate_count or \
                (current_count == merge_candidate_count and encoding_pair_candidate > merge_candidate):
                merge_candidate, merge_candidate_count = encoding_pair_candidate, current_count

        # merge_candidate, merge_candidate_count = cand_encoding_list[0]
        merge_history.append(merge_candidate)
        log.debug(f"merge_candidate {merge_candidate} count {merge_candidate_count}")

        next_merged_tokens: dict[tuple[bytes], int] = {}
        merged_tokens_cnt = 0

        for bytes_tuple, count in merged_tokens.items():
            new_bytes_list = []
            i = 0
            while i < len(bytes_tuple):
                if i < len(bytes_tuple) - 1 and (bytes_tuple[i], bytes_tuple[i + 1]) == merge_candidate:
                    new_bytes_list.append(bytes_tuple[i] + bytes_tuple[i + 1])
                    merged_tokens_cnt += count
                    i += 2
                else:
                    new_bytes_list.append(bytes_tuple[i])
                    i += 1
            next_merged_tokens[tuple(new_bytes_list)] = count
        
        # assert merged_tokens_cnt == merge_candidate_count, \
        #     f"Merge candidate {merge_candidate} merge candidate count {merge_candidate_count}, merged cnt {merged_tokens_cnt}"
        merged_tokens = next_merged_tokens
        current_encoding[next_encoding] = merge_candidate[0] + merge_candidate[1]
        next_encoding += 1
        log.debug(f"Completed merging {merge_candidate} into merged_tokens. Merged count {merged_tokens_cnt}.")
    
    return current_encoding, merge_history
