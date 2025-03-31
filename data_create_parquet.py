from argparse import ArgumentParser

import numpy as np
from datasets import DownloadConfig, load_dataset
from tqdm import trange

from transformers import AutoTokenizer


"""
Filter a datset by min token count and save to disk.
"""


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--dataset-path", type=str, default="allenai/dolmino-mix-1124")
    parser.add_argument("--dataset-names", type=str, default="flan")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--tokenizer", type=str, default="ibm-ai-platform/Bamba-9B")
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--thresholds", type=str, default="8192")
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--shard-size", type=int, default=2**18)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    def get_filter_fn(threshold: int):
        def filter_fn(examples):
            return [e > threshold for e in examples["toks"]]

        return filter_fn

    def get_chars_toks_batched(examples):
        c = [len(e) for e in examples["text"]]
        encodings = tokenizer(examples["text"], truncation=False, padding=False)
        t = [len(ids) for ids in encodings["input_ids"]]
        return {"chars": c, "toks": t}

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

        mapped_dataset = dataset.map(get_chars_toks_batched, batched=True, batch_size=args.batch_size)
        print(f"Num. examples, entire dataset: {len(dataset):.2E}")

        thresholds = [int(x) for x in args.thresholds.split(",")]
        for threshold in thresholds:
            print(f"Creating {threshold=}")
            filter_fn = get_filter_fn(threshold)
            mapped_dataset = mapped_dataset.filter(
                filter_fn, batched=True, batch_size=args.batch_size, num_proc=args.num_proc
            )
            print(f"Num. examples in threshold: {len(mapped_dataset):.2E}")
            n_toks = np.array(mapped_dataset["toks"]).sum()
            print(f"Tok in threshold (B): {n_toks / 1e9:.2f}")

            save_file_dir = (
                "".join(char if char.isalnum() else "_" for char in args.dataset_path)
                + f"/min_toks_{threshold}/"
                + "".join(char if char.isalnum() else "_" for char in dataset_name)
                + "/"
            )

            # https://github.com/huggingface/datasets/issues/7047#issuecomment-2233163406
            n_shards = (len(mapped_dataset) + args.shard_size - 1) // args.shard_size
            for shard_idx in trange(n_shards):
                filename = f"{shard_idx:05d}.parquet"
                mapped_dataset.shard(index=shard_idx, num_shards=n_shards, contiguous=True).to_parquet(
                    save_file_dir + filename
                )
            with open(save_file_dir + "n_toks.txt", "w") as f:
                f.write(str(n_toks))
