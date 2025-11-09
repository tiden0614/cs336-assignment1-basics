import logging
import os
import sys
from collections import defaultdict
from typing import BinaryIO

log = logging.getLogger("tokenizer")

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
# and then put words into a counting dictionary
async def word_count(file_name: str, start: int, end: int) -> dict[str, int]:
    pass


def initialize_merged_tokens(tokens: dict[str, int]) -> dict[tuple[bytes], int]:
    if "<|endoftext|>" not in tokens:
        tokens["<|endoftext|>"] = 1

    merged_tokens = {}
    for token, count in tokens.items():
        if token == "":
            pass

        if token == "<|endoftext|>":
            merged_tokens[(bytes(token, 'utf-8'),)] = count
            continue

        token_bytes_list = [bytes(c, 'utf-8') for c in token]
        merged_tokens[tuple(token_bytes_list)] = count
    
    return merged_tokens
    

def initialize_current_encoding() -> dict[bytes, int]:
    current_encoding = {i.to_bytes(1, 'big') : i for i in range(256)}
    current_encoding[bytes("<|endoftext|>", "utf-8")] = 256
    return current_encoding


# def concat_byte_seq(seq1: bytes, seq2: bytes) -> bytes:
#     concatenated = list(seq1) + list(seq2)




def tokenize(tokens: dict[str, int], passes: int) -> list[bytes, int]:
    # merged_tokens stores the partially compressed
    # tokens corresonding to their count
    # e.g.
    # {
    #   (l, o, w): 5,
    #   (w, i, d, e, st): 10,
    #   (l, o, w, e, st): 15,
    # }
    merged_tokens: dict[tuple[bytes], int] = initialize_merged_tokens(tokens)
    log.debug(f"Initial merged_tokens {merged_tokens}")

    # current_encoding stores the currently accepted
    # encodings.
    # it would look like something:
    # {
    #   l: 30,
    #   o: 31,
    #   ...
    #   st: 233,
    # }
    current_encoding: dict[bytes, int] = initialize_current_encoding()
    next_encoding = len(current_encoding)
    log.debug(f"Initial current_encoding {current_encoding}")

    for i in range(passes):
        log.info(f"Performing pass {i}. current_encoding_size {len(current_encoding)}")

        # candidate_encoding stores new encodings (that don't exist
        # in current_encoding) associate with the count they appear
        candidate_encoding: dict[bytes, int] = defaultdict(int)

        for bytes_tuple, count in merged_tokens.items():
            assert len(bytes_tuple) > 0, "empty byte tuple"

            if len(bytes_tuple) == 1:
                # when the whole bytes_tuple contains only one element, it must
                # have already been merged into the current_encoding
                assert bytes_tuple[0] in current_encoding.keys(), (bytes_tuple, current_encoding)
                continue

            for i in range(len(bytes_tuple) - 1, 0, -1):
                candidate_encoding[bytes_tuple[i - 1] + bytes_tuple[i]] += count
        
        # now we have collected all candidate encodings, sort them based on
        # the count and get the most frequent sequence
        log.info(f"Collected {len(candidate_encoding)} new candidates.")
        if len(candidate_encoding) == 0:
            log.info(f"No more candidates to merge.")
            break

        cand_encoding_list = list(candidate_encoding.items())

        # Sorting: we want to get the most frequent pair. If there
        # are two pairs whose frequencies are equal, break tie by
        # selecting the lexicographically greater one.
        cand_encoding_list.sort(reverse=True, key=lambda x: (x[1], x[0]))

        merge_candidate, merge_candidate_count = cand_encoding_list[0]
        log.info(f"merge_candidate {merge_candidate} count {merge_candidate_count}")

        next_merged_tokens: dict[tuple[bytes], int] = {}
        merged_tokens_cnt = 0

        for bytes_tuple, count in merged_tokens.items():
            new_bytes_list = []
            i = 0
            while i < len(bytes_tuple):
                if i < len(bytes_tuple) - 1 and bytes_tuple[i] + bytes_tuple[i + 1] == merge_candidate:
                    new_bytes_list.append(merge_candidate)
                    merged_tokens_cnt += count
                    i += 2
                else:
                    new_bytes_list.append(bytes_tuple[i])
                    i += 1
            next_merged_tokens[tuple(new_bytes_list)] = count
        
        assert merged_tokens_cnt == merge_candidate_count, \
            f"Merge candidate {merge_candidate} merge candidate count {merge_candidate_count}, merged cnt {merged_tokens_cnt}"
        merged_tokens = next_merged_tokens
        current_encoding[merge_candidate] = next_encoding
        next_encoding += 1
        log.info(f"Completed merging {merge_candidate} into merged_tokens. Merged count {merged_tokens_cnt}.")
    
    return list(current_encoding.items())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        log.warning("You need to specify the file to split")
        sys.exit(1)
    
    input_file_name = sys.argv[1]
    log.info(f"Splitting file {input_file_name}")
    
    with open(input_file_name, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token

