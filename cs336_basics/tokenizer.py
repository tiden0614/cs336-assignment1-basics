import concurrent.futures
import io
import logging
import os
import sys
import regex as re
import asyncio
from collections import defaultdict
from typing import BinaryIO, Generator

log = logging.getLogger("tokenizer")
TOKEN_SPLIT_PAT = re.compile(br"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
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
def word_count(
        file_name: str, 
        start: int, 
        end: int, 
        special_token_pattern: bytes) -> dict[str, int]:
    freq = defaultdict(int)
    with open(file_name, "rb") as file:
        with open('output3.log', 'w') as wfile:
            for token in produce_tokens(
                file, start, end, special_token_pattern, exclude_special_tokens=True):
                wfile.write(str(token) + '\n')
                freq[token] += 1
    return freq


def nested_token_iter_loop(
        text: bytes, 
        special_token_pattern: bytes, 
        exclude_special_tokens: bool) -> Generator[bytes, None, None]:
    if re.search(special_token_pattern, text):
        special_token_scanner = re.finditer(special_token_pattern, text)
        scanner_start_idx = 0
        for special_token in special_token_scanner:
            token_scanner = TOKEN_SPLIT_PAT.finditer(
                text, 
                pos=scanner_start_idx, 
                endpos=special_token.start())
            for token in token_scanner:
                yield token.group(0)

            if not exclude_special_tokens:
                yield special_token.group(0)

            scanner_start_idx = special_token.end()
        
        if scanner_start_idx < len(text):
            token_scanner = TOKEN_SPLIT_PAT.finditer(
                text, 
                pos=scanner_start_idx)
            for token in token_scanner:
                yield token.group(0)

    else:
        for token in TOKEN_SPLIT_PAT.finditer(text):
            yield token.group(0)


def produce_tokens(
        file: io.BufferedReader, 
        start: int, 
        end: int, 
        special_token_pattern: bytes,
        exclude_special_tokens: bool) -> Generator[bytes, None, None]:
    log.info(f"Starting to count tokens for file {file.name} start {start} end {end}")

    LOAD_SIZE = 512 << 10 # 512KB
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
    working_set: list[bytes] = [] # list of mini_chunks loaded so far
    chunks_loaded = 0
    token_count = 0
    total_load_size = end - start

    def working_set_size():
        return sum(map(len, working_set))

    while loaded_size < total_load_size or working_set_size() != 0:
        if chunks_loaded > 0 and chunks_loaded % 100 == 0:
            log.info(f"Loaded {chunks_loaded} chunks ({loaded_size >> 10}KB). "
                        f"Current token count {token_count}")

        load_size = min(LOAD_SIZE, total_load_size - loaded_size)
        log.debug("Loading %dMB into memory", loaded_size >> 20)
        mini_chunk = b""
        if loaded_size < total_load_size:
            mini_chunk = file.read(load_size)
            chunks_loaded += 1
            loaded_size += load_size

        pre_split = None

        special_token_matcher = re.search(special_token_pattern, mini_chunk)
        if special_token_matcher:
            log.debug("Found mini_chunk containing special token. "\
                    "chunks_loaded=%d mini_chunk=%d found_at=%d special_token=%s",
                    chunks_loaded, len(mini_chunk), special_token_matcher.start(), 
                    special_token_matcher.group(0))

            pre_split = mini_chunk[special_token_matcher.end():]

            if exclude_special_tokens:
                working_set.append(mini_chunk[:special_token_matcher.start()])
            else:
                working_set.append(mini_chunk[:special_token_matcher.end()])

            log.debug(
                "Splitting mini_chunk %d, %d, %d", 
                len(mini_chunk), special_token_matcher.start(), len(pre_split))
        else:
            log.debug("can't find special token splitter. enqueuing mini_chunk")
            working_set.append(mini_chunk)
            if loaded_size < total_load_size:
                log.debug("Special token not found. Appending to working set and continue to load.")
                continue

        to_word_count: bytes = b"".join(working_set)
        working_set.clear()

        for token in nested_token_iter_loop(
            to_word_count, special_token_pattern, exclude_special_tokens):
            token_count += 1
            yield token

        if pre_split is not None:
            working_set.append(pre_split)
    
    log.info(f"Finished portion. loaded_size={loaded_size} chunks_loaded={chunks_loaded} "\
                f"tokens_count={token_count}")


def validate_split_boundaries(parallelism: int, boundaries: list[int])-> list[tuple[int, int]]:
    assert len(boundaries) > 0

    result: list[tuple[int, int]] = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        assert start < end, f"Found invalid start end pair {start},{end} at {i} for {boundaries}"
        result.append((start, end))
    
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


def build_special_tokens_pattern(special_tokens: list[str]) -> bytes:
    return "|".join(map(re.escape, sorted(special_tokens, reverse=True))).encode('utf-8')


def pre_tokenize(file_name: str, parallelism: int, special_tokens: list[str]) -> dict[str, int]:
    # 0. build special tokens regex
    # 1. find boundaries
    # 2. for each boundary pair, assign a thread to do word count
    # 3. wait for all threads to finish, and then merge all outputs

    # Step 0: build special tokens regex
    log.info(f"Received special tokens {special_tokens}")
    special_tokens_pattern = build_special_tokens_pattern(special_tokens)
    log.info(f"Using special tokens pattern {special_tokens_pattern}")

    # Step 1: find boundaries
    log.info(f"Splitting file {file_name} into {parallelism} chunks")
    with open(file_name, "rb") as f:
        boundaries = find_chunk_boundaries(f, parallelism, b"<|endoftext|>")
    
    # Double check the correctness of the split output
    boundary_pairs = validate_split_boundaries(parallelism, boundaries)
    parallelism = len(boundary_pairs)

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


def get_bytes_pair(token_sig):
    for i in range(len(token_sig) - 1):
        yield token_sig[i], token_sig[i + 1]


def replace_bytes_pair(merged: list[bytes], b1: bytes, b2: bytes):
    found_match = True
    while found_match:
        found_match = False
        for i in range(len(merged) - 1):
            ob1, ob2 = merged[i], merged[i + 1]
            if ob1 == b1 and ob2 == b2:
                found_match = True
                # delete ob1 and ob2 from merged and then insert b1 + b2
                del merged[i : i + 2]
                merged.insert(i, b1 + b2)
                break


def initialize_merged_token_idx(
        merged_tokens: dict[tuple[bytes], int]) -> dict[tuple[bytes, bytes], set[tuple[bytes]]]:
    merged_tokens_idx = defaultdict(set)

    for token_sig in merged_tokens.keys():
        for b1, b2 in get_bytes_pair(token_sig):
            merged_tokens_idx[(b1, b2)].add(token_sig)

    return merged_tokens_idx


def merge_bytes_pair_for_token(token: tuple[bytes], merge_candidate: tuple[bytes, bytes]) -> tuple[bytes]:
    new_token_sig_list = []
    i = 0
    while i < len(token):
        if i < len(token) - 1 and (token[i], token[i + 1]) == merge_candidate:
            new_token_sig_list.append(token[i] + token[i + 1])
            i += 2
        else:
            new_token_sig_list.append(token[i])
            i += 1
    
    return tuple(new_token_sig_list)


def learn_merges(
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
    merged_tokens_idx: dict[tuple[bytes, bytes], set[tuple[bytes]]] = \
        initialize_merged_token_idx(merged_tokens)
    log.debug("Initial merged_tokens %s", merged_tokens)

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
    log.debug("Initial current_encoding %s", current_encoding)

    merge_history: list[tuple[bytes, bytes]] = []

    for i in range(passes):
        if i > 0 and i % 200 == 0:
            log.info(f"Performing pass {i}. current_encoding_size {len(current_encoding)}")

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

        merge_history.append(merge_candidate)

        merged_tokens_cnt = 0

        # Merge:
        # 1. look up merge_tokens_idx and find relevant tokens to modify
        # 2. get the new_token_sig by merging bytes together
        # 3. replace the old_token_sig with new_token_sig
        # 4. remove idx references to old_token_sig
        # 5. add idx references to new_token_sig

        # We need the copy here because we are going to mutate the idx entry
        relevant_merge_tokens: set[tuple[bytes]] = merged_tokens_idx[merge_candidate].copy()

        for old_token_sig in relevant_merge_tokens:
            new_token_sig = merge_bytes_pair_for_token(old_token_sig, merge_candidate)
            
            # replace old_token_sig in merged_tokens
            assert new_token_sig not in merged_tokens, (new_token_sig, merged_tokens)
            merged_tokens[new_token_sig] = merged_tokens[old_token_sig]
            del merged_tokens[old_token_sig]

            # remove idx references to any byte pairs in the old_token_sig
            for b1, b2 in get_bytes_pair(old_token_sig):
                merged_tokens_idx[(b1, b2)].discard(old_token_sig)
            
            # add idx references to any byte pairs in the new_token_sig
            for b1, b2 in get_bytes_pair(new_token_sig):
                merged_tokens_idx[(b1, b2)].add(new_token_sig)
            
            merged_tokens_cnt += 1

        current_encoding[next_encoding] = merge_candidate[0] + merge_candidate[1]
        next_encoding += 1
    
    log.info(f"Finished training bpe. Vocab size {len(current_encoding)}")
    return current_encoding, merge_history


class Tokenizer:
    def __init__(
            self, 
            vocab: dict[int, bytes], 
            merges: list[tuple[bytes, bytes]], 
            special_tokens: list[str] = None):
        self.vocab = vocab
        self.vocab_reverse = {byte_rep: encoded for encoded, byte_rep in self.vocab.items()}
        self.merges = merges
        self.merges_set = set(merges)
        self.special_tokens = special_tokens or ["<|endoftext|>"]
    
    @classmethod
    def from_files(
        clz, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] = None):
        raise Exception("Unimplemented")
    
    def decode(self, ids: list[int]) -> str:
        working_set: list[str] = []

        for id in ids:
            working_set.append(self.vocab[id])

        return b"".join(working_set).decode('utf-8')
    
    def encode(self, text: str) -> list[int]:
        result = []
        for token in self._tokenize_text(text):
            for encoded in self._encode_token(token):
                result.append(encoded)
        return result
    
    def encode_iterable(self, file: io.BufferedReader) -> Generator[int, None, None]:
        special_token_pattern = build_special_tokens_pattern(self.special_tokens)
        file_size = file.tell()
        for token in produce_tokens(
            file, 0, file_size, special_token_pattern, exclude_special_tokens=False):
            for encoded in self._encode_token(token):
                yield encoded
    
    def _encode_token(self, token: bytes) -> Generator[int, None, None]:
        if len(token) == 0:
            raise Exception("empty token")
        
        if len(token) == 1:
            yield self.vocab_reverse[token]
            return
        
        full_encoding = self.vocab_reverse.get(token)
        if full_encoding:
            yield full_encoding
            return

        merged = [t.to_bytes(1, 'big') for t in token]

        found_match = True
        while found_match:
            found_match = False
            for mb1, mb2 in self.merges:
                for b1, b2 in get_bytes_pair(merged):
                    if mb1 == b1 and mb2 == b2:
                        found_match = True
                        replace_bytes_pair(merged, b1, b2)
                        break
                if found_match:
                    break
        
        for b in merged:
            yield self.vocab_reverse[b]
                        

    def _encode_token_stack(self, token: bytes) -> Generator[int, None, None]:
        if len(token) == 0:
            raise Exception("empty token")
        
        if len(token) == 1:
            yield self.vocab_reverse[token]
            return
        
        full_encoding = self.vocab_reverse.get(token)
        if full_encoding:
            yield full_encoding
            return
        
        merged = []
        pending = [token[i].to_bytes(1, 'big') for i in range(len(token) - 1, -1, -1)]
        while pending:
            if len(merged) == 0:
                merged.append(pending.pop())
                continue

            b1, b2 = merged[-1], pending.pop()
            if (b1, b2) in self.merges_set:
                merged.pop()
                pending.append(b1 + b2)

                while merged:
                    pending.append(merged.pop())
            else:
                merged.append(b2)
        
        for b in merged:
            yield self.vocab_reverse[b]
    
    def _tokenize_text(self, text: str) -> Generator[bytes, None, None]:
        for token in nested_token_iter_loop(
            text.encode('utf-8'), 
            build_special_tokens_pattern(self.special_tokens), 
            exclude_special_tokens=False):
            yield token


        
