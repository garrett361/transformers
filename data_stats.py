from argparse import ArgumentParser

from datasets import DownloadConfig, load_dataset

from transformers import AutoTokenizer


def get_chars_toks_batched(examples):
    c = [len(e) for e in examples["text"]]
    encodings = tokenizer(examples["text"], truncation=False, padding=False)
    t = [len(ids) for ids in encodings["input_ids"]]
    return {"chars": c, "toks": t}


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--dataset-path", type=str, default="allenai/dolmino-mix-1124")
    parser.add_argument("--dataset-name", type=str, default="pes2o")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--tokenizer", type=str, default="ibm-ai-platform/Bamba-9B")
    parser.add_argument("--num-examples", type=int, default=None)
    args = parser.parse_args()

    dataset = load_dataset(
        args.dataset_path,
        args.dataset_name,
        split=args.dataset_split,
        download_config=DownloadConfig(resume_download=True, num_proc=8),
    )
    print(f"Num. examples, entire dataset: {len(dataset):.2E}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if args.num_examples is not None:
        mapped_dataset = dataset.select(range(len(dataset) - args.num_examples, len(dataset) - 1)).map(
            get_chars_toks_batched, batched=True, batch_size=8192
        )
    else:
        mapped_dataset = dataset.map(get_chars_toks_batched, batched=True, batch_size=8192)
    mapped_dataset.set_format("numpy")
    print(f"Num. examples analyzed: {len(mapped_dataset):.2E}")

    chars = mapped_dataset["chars"]
    toks = mapped_dataset["toks"]
    print(f"{toks.mean()=}")
    print(f"{toks.max()=}")
    print(f"Chars/Tok: {chars.sum() / toks.sum():.2f}")

    tok_thresholds = [
        2046,
        4096,
        8192,
    ]
    for tok_t in tok_thresholds:
        print(f"Num. examples longer than {tok_t}: {sum(toks > tok_t)}")
