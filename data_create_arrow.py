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


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--dataset-path", type=str, default="allenai/dolmino-mix-1124")
    parser.add_argument("--dataset-names", type=str, default="flan")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--tokenizer", type=str, default="ibm-ai-platform/Bamba-9B")
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--min_toks", type=int, default=8192)
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--mib", type=int, default=128)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    def get_toks(examples):
        tokens = tokenizer(
            examples["text"], truncation=False, padding=False, return_attention_mask=False, return_token_type_ids=False
        )["input_ids"]
        return tokens

    def get_toks_dict(examples):
        tokens = get_toks(examples)
        n_toks = [len(ids) for ids in tokens]
        return {"n_toks": n_toks, "tokens": tokens}

    def get_filter_fn(min_toks: int):
        def filter_fn(examples):
            return [e > min_toks for e in examples["n_toks"]]

        return filter_fn

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

        print(f"Num. examples, entire dataset: {len(dataset):.2E}")

        dataset = dataset.map(
            get_toks_dict,
            batched=True,
            batch_size=args.batch_size,
            num_proc=args.num_proc,
            remove_columns=dataset.column_names,
        )
        filter_fn = get_filter_fn(args.min_toks)
        dataset = dataset.filter(filter_fn, batched=True, batch_size=args.batch_size, num_proc=args.num_proc)

        print(f"Num. examples in min_toks: {len(dataset):.2E}")

        n_toks = np.array(dataset["n_toks"]).sum()

        save_file_dir = Path(
            "".join(char if char.isalnum() else "_" for char in args.dataset_path)
            + f"/min_toks_{args.min_toks}/"
            + "".join(char if char.isalnum() else "_" for char in dataset_name)
            + "/"
        )
        save_file_dir.mkdir(parents=True, exist_ok=True)

        # "tokens" is an arbitrary header. You can use any header, and simply update config.col_name above to match
        schema = pa.schema([pa.field("tokens", pa.uint32())])

        tokens = dataset["tokens"]
        n_toks = dataset["n_toks"]
        total_bytes = 4 * sum(n_toks)
        data_idx = 0
        max_data_idx = len(tokens) - 1
        shard_idx = 0
        max_bytes = 2**20 * args.mib
        with tqdm(total=total_bytes) as pbar:
            while True:
                shard_filename = f"{shard_idx:05d}.arrow"
                curr_bytes = 0
                shard_idx += 1
                with pa.ipc.new_file(save_file_dir.joinpath(shard_filename), schema) as writer:
                    while data_idx <= max_data_idx and curr_bytes < max_bytes:
                        writer.write(pa.record_batch([tokens[data_idx]], schema=schema))
                        tok_bytes = 4 * n_toks[data_idx]
                        curr_bytes += tok_bytes
                        data_idx += 1
                        if curr_bytes >= max_bytes:
                            pbar.update(curr_bytes)
                    if data_idx > max_data_idx:
                        break

        with open(save_file_dir.joinpath("n_toks.txt"), "w") as f:
            f.write(str(n_toks))
