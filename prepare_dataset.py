"""
prepare_dataset.py
──────────────────
Download FineWeb-Edu sample-10BT, tokenize with the Mixtral tokenizer,
and write a Megatron-LM indexed dataset (.bin / .idx) of ~100M tokens.

Run once before the benchmark, on a login node or in a short pre-job:

    python prepare_dataset.py \
        --output_dir  <workspace>/datasets/fineweb_edu_mixtral_100M \
        --target_tokens 100_000_000 \
        --seq_len 2048 \
        --seed 1234

Requirements (install in your conda env):
    pip install datasets transformers numpy

The Megatron indexed-dataset writer used here is a self-contained
reimplementation of Megatron's tools/preprocess_data.py logic so we do
not need a full Megatron checkout just for dataset prep.
"""

import argparse
import os
import struct
import time
import numpy as np
from pathlib import Path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str,
                   default="datasets/fineweb_edu_mixtral_100M")
    p.add_argument("--dataset_name", type=str,
                   default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset_config", type=str, default="sample-10BT")
    p.add_argument("--model_name", type=str,
                   default="mistralai/Mixtral-8x7B-v0.1")
    p.add_argument("--target_tokens", type=int, default=100_000_000,
                   help="Stop after collecting this many tokens")
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--num_proc", type=int, default=16,
                   help="Parallel tokenization workers")
    p.add_argument("--hf_cache_dir", type=str, default=None,
                   help="HuggingFace cache dir (leave None for default)")
    return p.parse_args()


# ── Megatron indexed-dataset writer ──────────────────────────────────────────
# Writes the binary format that Megatron-Core's MMapIndexedDataset can read.
# Format spec:
#   .idx  header + per-document (dtype, size, pointer) table
#   .bin  raw token ids packed as uint16 (vocab ≤ 65535) or uint32

HDR_MAGIC  = b"MMIDIDX\x00\x00"   # 9 bytes
HDR_VERSION = struct.pack("<Q", 1) # 8 bytes  → uint64
DTYPE_CODE  = 2                    # uint16; Mixtral vocab=32000 fits in uint16

class MegatronIndexedDatasetWriter:
    """Minimal writer for Megatron MMapIndexedDataset v1."""

    def __init__(self, prefix: str):
        self.bin_path = prefix + ".bin"
        self.idx_path = prefix + ".idx"
        self._bin = open(self.bin_path, "wb")
        self._sizes   = []   # number of tokens per document
        self._pointers = []  # byte offset of each document in .bin
        self._ptr = 0

    def add_document(self, tokens: np.ndarray):
        """tokens: 1-D numpy array of dtype uint16."""
        data = tokens.astype(np.uint16).tobytes()
        self._bin.write(data)
        self._pointers.append(self._ptr)
        self._sizes.append(len(tokens))
        self._ptr += len(data)

    def finalize(self):
        self._bin.close()
        n_docs = len(self._sizes)
        with open(self.idx_path, "wb") as f:
            f.write(HDR_MAGIC)
            f.write(HDR_VERSION)
            f.write(struct.pack("<B", DTYPE_CODE))   # dtype code
            f.write(struct.pack("<q", n_docs))        # n documents  (int64)
            f.write(struct.pack("<q", n_docs))        # n sequences  (int64, same here)
            # sizes  (int32 array)
            f.write(np.array(self._sizes,    dtype=np.int32).tobytes())
            # pointers (int64 array)
            f.write(np.array(self._pointers, dtype=np.int64).tobytes())
        print(f"  Wrote {n_docs:,} documents → {self.bin_path}")
        print(f"  Index → {self.idx_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(out_dir / "fineweb_edu_mixtral_100M_text_document")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    print(f"Loading tokenizer from {args.model_name} ...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        cache_dir=args.hf_cache_dir,
    )
    # Mixtral tokenizer does not add BOS by default in fast mode; be explicit.
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = True   # add EOS so documents are delimited

    # ── Dataset ──────────────────────────────────────────────────────────────
    print(f"Streaming {args.dataset_name} / {args.dataset_config} ...")
    from datasets import load_dataset
    ds = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split="train",
        streaming=True,
        cache_dir=args.hf_cache_dir,
    )

    # Shuffle the stream with a buffer (reproducible via seed)
    ds = ds.shuffle(seed=args.seed, buffer_size=10_000)

    # ── Tokenize + pack + write ───────────────────────────────────────────────
    writer = MegatronIndexedDatasetWriter(prefix)
    total_tokens = 0
    doc_count    = 0
    t0 = time.time()

    # We pack tokens into fixed-length sequences of args.seq_len.
    # Leftover tokens at the end of a document are carried over to the next.
    carry = []

    print(f"Tokenizing until {args.target_tokens:,} tokens ...")
    for example in ds:
        text = example.get("text", "")
        if not text:
            continue

        ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        carry.extend(ids)

        # Emit as many full sequences as we have
        while len(carry) >= args.seq_len:
            seq = np.array(carry[:args.seq_len], dtype=np.uint16)
            writer.add_document(seq)
            carry = carry[args.seq_len:]
            total_tokens += args.seq_len
            doc_count    += 1

            if total_tokens >= args.target_tokens:
                break

        if total_tokens >= args.target_tokens:
            break

        if doc_count % 5_000 == 0 and doc_count > 0:
            elapsed = time.time() - t0
            rate = total_tokens / elapsed
            print(f"  {total_tokens/1e6:.1f}M / {args.target_tokens/1e6:.0f}M tokens "
                  f"| {rate/1e3:.1f}k tok/s | {doc_count:,} docs")

    # Flush any remaining carry as a final (possibly shorter) document
    if carry and total_tokens < args.target_tokens:
        writer.add_document(np.array(carry, dtype=np.uint16))
        total_tokens += len(carry)

    writer.finalize()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"Total tokens : {total_tokens:,}")
    print(f"Total docs   : {doc_count:,}")
    print(f"Output prefix: {prefix}")
    print(f"\nUse in Megatron with:")
    print(f"  --data-path {prefix}")


if __name__ == "__main__":
    main()
