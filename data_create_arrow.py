import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pyarrow as pa
from datasets import DownloadConfig, load_dataset
from tqdm import tqdm

from transformers import AutoTokenizer


"""
Filter a datset by min token count and save to disk.

https://github.com/foundation-model-stack/bamba/tree/main/training/data#training-on-your-own-data
"""

# Actual chars per token is slightly higher, but we want to be conservative here.
CHAR_PER_TOKEN = 4
BYTES_PER_TOKEN = 4
BYTES_PER_MiB = 2**20


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--dataset-path", type=str, default="allenai/dolmino-mix-1124")
    parser.add_argument("--dataset-names", type=str, default="flan")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--tokenizer", type=str, default="ibm-ai-platform/Bamba-9B")
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--min_toks", type=int, default=8192)
    parser.add_argument("--max_toks", type=int, default=None)
    parser.add_argument("--num-proc", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--mib", type=int, default=128)
    args = parser.parse_args()
    if args.num_proc is None:
        args.num_proc = os.cpu_count() // 2

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    def map_fn(examples):
        # Perform a rough filtering according to approx char count
        if args.max_toks is None:
            filtered_text = [t for t in examples["text"] if len(t) // CHAR_PER_TOKEN >= args.min_toks]
        else:
            filtered_text = [t for t in examples["text"] if args.max_toks >= len(t) // CHAR_PER_TOKEN >= args.min_toks]
        if not filtered_text:
            return {"n_toks": [], "tokens": []}
        tokens = tokenizer(
            filtered_text,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        if args.max_toks is None:
            tokens = [t for t in tokens if len(t) >= args.min_toks]
        else:
            tokens = [t for t in tokens if args.max_toks >= len(t) >= args.min_toks]
        n_toks = [len(t) for t in tokens]
        return {"n_toks": n_toks, "tokens": tokens}

    dataset_names = args.dataset_names.split(",")
    for dataset_name in dataset_names:
        print(f"\n***Processing {args.dataset_path}:{dataset_name}***\n")
        dataset = load_dataset(
            args.dataset_path,
            dataset_name,
            split=args.dataset_split,
            download_config=DownloadConfig(resume_download=True, num_proc=args.num_proc),
        )
        if args.num_examples is not None:
            # Just for quick testing
            dataset = dataset.select(range(args.num_examples))

        print(f"Num. examples entire dataset: {len(dataset):.2E}")

        dataset = dataset.map(
            map_fn,
            batched=True,
            batch_size=args.batch_size,
            num_proc=args.num_proc,
            remove_columns=dataset.column_names,
        )
        conds = f"toks >= {args.min_toks}"
        if args.max_toks is not None:
            conds = f"{args.max_toks} >= " + conds
        print(f"Num. examples with {conds}: {len(dataset):.2E}")
        n_toks_np = np.array(dataset["n_toks"])
        total_toks = n_toks_np.sum().item()
        print(f"Num. tokens (B) with {conds}: {total_toks / 1e9}")
        cache_dir = os.getenv("HF_CACHE", "~/.cache/huggingface/datasets/")
        save_file_dir = Path(
            cache_dir
            + "".join(char if char.isalnum() else "_" for char in args.dataset_path)
            + (
                f"/min_toks_{args.min_toks}/"
                if args.max_toks is None
                else f"/min_toks_{args.min_toks}_max_toks_{args.max_toks}/"
            )
            + "".join(char if char.isalnum() else "_" for char in dataset_name)
            + "/"
        ).expanduser()
        save_file_dir.mkdir(parents=True, exist_ok=True)

        # "tokens" is expected by `fms-fsdp`.
        schema = pa.schema([pa.field("tokens", pa.uint32())])
        print(f"Writing to {save_file_dir=}")

        max_bytes = BYTES_PER_MiB * args.mib
        total_bytes = BYTES_PER_TOKEN * total_toks
        n_expected_shards = (total_bytes + max_bytes - 1) // max_bytes
        data_iter = dataset.iter(batch_size=1)
        shard_idx = 0
        done = False
        with tqdm(total=n_expected_shards) as pbar:
            while not done:
                shard_filename = f"{shard_idx:05d}.arrow"
                curr_bytes = 0
                shard_idx += 1
                with pa.ipc.new_file(save_file_dir.joinpath(shard_filename), schema) as writer:
                    while curr_bytes < max_bytes:
                        try:
                            data = next(data_iter)
                        except StopIteration:
                            pbar.update(1)
                            done = True
                            break
                        tokens = data["tokens"]
                        n_toks = data["n_toks"][0]
                        writer.write(pa.record_batch(tokens, schema=schema))
                        curr_bytes += BYTES_PER_TOKEN * n_toks
                        if curr_bytes >= max_bytes:
                            pbar.update(1)

        with open(save_file_dir.joinpath("tokens_sum.txt"), "w") as f:
            f.write(str(n_toks_np.sum().item()))
        with open(save_file_dir.joinpath("tokens_mean.txt"), "w") as f:
            f.write(str(n_toks_np.mean().item()))
        with open(save_file_dir.joinpath("tokens_std.txt"), "w") as f:
            f.write(str(n_toks_np.std().item()))
        with open(save_file_dir.joinpath("tokens_median.txt"), "w") as f:
            f.write(str(np.median(n_toks_np).item()))
        with open(save_file_dir.joinpath("tokens_max.txt"), "w") as f:
            f.write(str(n_toks_np.max().item()))
