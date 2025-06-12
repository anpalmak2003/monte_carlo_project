# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming dataset conversion scripts for C4 and The Pile."""

import os
import platform
import warnings
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Optional, Union

import numpy as np
from streaming import MDSWriter
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

import os
import json
from typing import Dict, Iterable, Optional, Union
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase

class ConcatMode(Enum):
    NO_CONCAT = "NO_CONCAT"
    CONCAT_TOKENS = "CONCAT_TOKENS"


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(description="Convert dataset into MDS format, optionally concatenating and tokenizing")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("--data_subset", type=str, default=None, help='E.g. "all" or "en"')
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--compression", type=str, default=None)

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--concat_tokens", type=int, help="Convert text to tokens and concatenate up to this many tokens"
    )

    parser.add_argument("--tokenizer", type=str, required=False, default=None)
    parser.add_argument("--bos_text", type=str, required=False, default=None)
    parser.add_argument("--eos_text", type=str, required=False, default=None)
    parser.add_argument("--no_wrap", default=False, action="store_true")

    parsed = parser.parse_args()

    if os.path.isdir(parsed.out_root) and len(set(os.listdir(parsed.out_root)).intersection(set(parsed.splits))) > 0:
        raise ValueError(
            f"--out_root={parsed.out_root} contains {os.listdir(parsed.out_root)} which cannot overlap with the requested splits {parsed.splits}."
        )

    # Make sure we have needed concat options
    if parsed.concat_tokens is not None and isinstance(parsed.concat_tokens, int) and parsed.tokenizer is None:
        parser.error("When setting --concat_tokens, you must specify a --tokenizer")

    # now that we have validated them, change BOS/EOS to strings
    if parsed.bos_text is None:
        parsed.bos_text = ""
    if parsed.eos_text is None:
        parsed.eos_text = ""
    return parsed


@dataclass
class DataSplitConstants:
    hf_split: str
    folder_split: str
    raw_samples: int
    truncated_samples: Union[int, None]


@dataclass
class DatasetConstants:
    chars_per_sample: int
    chars_per_token: int
    splits = {}

    def __iter__(self):
        for _, v in self.splits.items():
            yield v


class TrainSmallConstants(DataSplitConstants):
    def __init__(
        self,
        hf_split: str = "train",
        folder_split: str = "train_small",
        raw_samples: int = 1000000,
        truncated_samples: int = 100000,
    ):
        super().__init__(hf_split, folder_split, raw_samples, truncated_samples)


class ValSmallConstants(DataSplitConstants):
    def __init__(
        self,
        hf_split: str = "validation",
        folder_split: str = "val_small",
        raw_samples: int = 10000,
        truncated_samples: int = 10000,
    ):
        super().__init__(hf_split, folder_split, raw_samples, truncated_samples)


pileconstants = DatasetConstants(
    chars_per_sample=6212,  # Computed over validation set
    chars_per_token=4,  # OpenAI estimate
)
pileconstants.splits["train"] = DataSplitConstants(
    hf_split="train", folder_split="train", raw_samples=210607728, truncated_samples=None
)
pileconstants.splits["train_small"] = DataSplitConstants(
    hf_split="train", folder_split="train_small", raw_samples=1000000, truncated_samples=100000
)
pileconstants.splits["val"] = DataSplitConstants(
    hf_split="validation", folder_split="val", raw_samples=214670, truncated_samples=None
)
pileconstants.splits["val_small"] = DataSplitConstants(
    hf_split="validation", folder_split="val_small", raw_samples=10000, truncated_samples=10000
)

c4constants = DatasetConstants(
    chars_per_sample=2163,  # Computed over validation set
    chars_per_token=4,  # OpenAI estimate
)
c4constants.splits["train"] = DataSplitConstants(
    hf_split="train", folder_split="train", raw_samples=364868892, truncated_samples=None
)
c4constants.splits["train_small"] = DataSplitConstants(
    hf_split="train", folder_split="train_small", raw_samples=1000000, truncated_samples=100000
)
c4constants.splits["val"] = DataSplitConstants(
    hf_split="validation", folder_split="val", raw_samples=364608, truncated_samples=None
)
c4constants.splits["val_small"] = DataSplitConstants(
    hf_split="validation", folder_split="val_small", raw_samples=10000, truncated_samples=10000
)

CONSTS = {"c4": c4constants, "the_pile": pileconstants}


def read_jsonl_files_from_folder(folder_path: str) -> Iterable[Dict[str, str]]:
    """Рекурсивно читает все JSONL-файлы из папки и её подпапок."""
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.jsonl'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        yield json.loads(line)

class NoConcatDataset(IterableDataset):
    """An IterableDataset that returns text samples for MDSWriter.

    Returns dicts of {'text': bytes}
    """

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def __iter__(self) -> Iterable[Dict[str, bytes]]:
        for sample in read_jsonl_files_from_folder(self.dataset_path):
            # convert to bytes to store in MDS binary format
            yield {"text": sample["text"].encode("utf-8")}


class ConcatTokensDataset(IterableDataset):
    """An IterableDataset that returns token samples for MDSWriter.

    Returns dicts of {'tokens': bytes}
    """

    def __init__(
        self,
        dataset_path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        bos_text: str,
        eos_text: str,
        no_wrap: bool,
    ):
        self.tokenizer = tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.max_length = max_length
        self.bos_text = bos_text
        self.eos_text = eos_text
        self.should_wrap = not no_wrap
        self.dataset_path = dataset_path

        self.bos_tokens = self.tokenizer(self.bos_text, truncation=False, padding=False, add_special_tokens=False)[
            "input_ids"
        ]
        if len(self.bos_tokens) > 1:
            warnings.warn(
                f"You specified --concat_tokens with --bos_text, but your BOS text is not tokenizing to one token\
                , instead we got {self.bos_tokens}. Quit if this was in error."
            )

        self.eos_tokens = self.tokenizer(self.eos_text, truncation=False, padding=False, add_special_tokens=False)[
            "input_ids"
        ]
        if len(self.eos_tokens) > 1:
            warnings.warn(
                f"You specified --concat_tokens with --eos_text, but your EOS text is not tokenizing to one token\
                , instead we got {self.eos_tokens}. Quit if this was in error."
            )

    def __iter__(self) -> Iterable[Dict[str, bytes]]:
        buffer = []
        for sample in read_jsonl_files_from_folder(self.dataset_path):
            encoded = self.tokenizer(sample["text"], truncation=False, padding=False)
            iids = encoded["input_ids"]
            buffer = buffer + self.bos_tokens + iids + self.eos_tokens
            while len(buffer) >= self.max_length:
                concat_sample = buffer[: self.max_length]
                buffer = buffer[self.max_length :] if self.should_wrap else []
                yield {
                    # convert to bytes to store in MDS binary format
                    "tokens": np.asarray(concat_sample).tobytes()
                }


def build_hf_dataset(
    dataset_path: str,
    mode: ConcatMode,
    max_length: int,
    bos_text: Optional[str],
    eos_text: Optional[str],
    no_wrap: bool,
    tokenizer: Optional[PreTrainedTokenizerBase],
) -> IterableDataset:
    """Build an IterableDataset over the local dataset source data.

    Args:
        dataset_path (str): Path to the dataset folder.
        mode (ConcatMode): NO_CONCAT, or CONCAT_TOKENS
        bos_text (str): text to insert at the beginning of each sequence
        eos_text (str): text to insert at the end of each sequence
        no_wrap (bool): if concatenating, whether to wrap text across `max_length` boundaries
        tokenizer (PreTrainedTokenizerBase): if mode is CONCAT_TOKENS, the tokenizer to use

    Returns:
        An IterableDataset.
    """
    if mode == ConcatMode.NO_CONCAT:
        dataset = NoConcatDataset(dataset_path=dataset_path)
    else:
        assert bos_text is not None
        assert eos_text is not None
        assert tokenizer is not None
        if bos_text + eos_text == "":
            test_tokens = tokenizer("test")
            if (
                test_tokens["input_ids"][0] != tokenizer.bos_token_id
                and test_tokens["input_ids"][-1] != tokenizer.eos_token_id
            ):
                tok_error_msg = "This tokenizer does not insert an EOS nor BOS token. "
                tok_error_msg += "Concatenating with this tokenizer will result in sequences being "
                tok_error_msg += "attached without a separating token. Please use another tokenizer, "
                tok_error_msg += "such as facebook/opt-125m, or specify EOS/BOS text with e.g. "
                tok_error_msg += "--bos_text=<|endoftext|>."
                raise ValueError(tok_error_msg)
        dataset = ConcatTokensDataset(
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            max_length=max_length,
            bos_text=bos_text,
            eos_text=eos_text,
            no_wrap=no_wrap,
        )
    return dataset


def _est_progress_denominator(
    total_samples: int, chars_per_sample: int, chars_per_token: int, mode: ConcatMode, max_length: int
):
    est_tokens_per_sample = chars_per_sample // chars_per_token
    if mode == ConcatMode.NO_CONCAT:
        return total_samples
    elif mode == ConcatMode.CONCAT_TOKENS:
        return total_samples * est_tokens_per_sample // max_length


def build_dataloader(dataset, batch_size) -> DataLoader:
    # Multiple workers is only supported on linux machines
    if "linux" in platform.platform().lower():
        num_workers = min(64, 1)  # type: ignore
    else:
        num_workers = 0

    # If using multiple workers, configure each worker to prefetch as many samples as it can, up to
    # the aggregate device batch size
    # If not using workers, the torch DataLoader expects the default value for prefetch_factor,
    # which non-intuitively must be 2.
    prefetch_factor = max(1, 2 * batch_size // num_workers) if num_workers > 0 else 2

    return DataLoader(
        dataset=dataset,
        sampler=None,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )


def generate_samples(loader: DataLoader, truncate_num_samples: Optional[int] = None) -> Iterable[Dict[str, bytes]]:
    """Generator over samples of a dataloader.

    Args:
       loader (DataLoader): A dataloader emitting batches like {key: [sample0_bytes, sample1_bytes, sample2_bytes, ...]}
       truncate_num_samples (Optional[int]): An optional # of samples to stop at.

    Yields:
        Sample dicts.
    """
    n_samples = 0
    for batch in loader:
        keys = list(batch.keys())
        current_bs = len(batch[keys[0]])
        for idx in range(current_bs):
            if truncate_num_samples is not None and n_samples == truncate_num_samples:
                return
            n_samples += 1
            yield {k: v[idx] for k, v in batch.items()}


# Добавляем стандартные константы для датасетов, которые не указаны в CONSTS
DEFAULT_CONSTS = DatasetConstants(
    chars_per_sample=2163,  # Примерное значение, можно изменить
    chars_per_token=4,  # Примерное значение, можно изменить
)
DEFAULT_CONSTS.splits["train"] = DataSplitConstants(
    hf_split="train", folder_split="train", raw_samples=1000000, truncated_samples=None
)
DEFAULT_CONSTS.splits["train_small"] = DataSplitConstants(
    hf_split="train", folder_split="train_small", raw_samples=500000, truncated_samples=500000
)
DEFAULT_CONSTS.splits["val"] = DataSplitConstants(
    hf_split="validation", folder_split="val", raw_samples=200000, truncated_samples=None
)
DEFAULT_CONSTS.splits["val_small"] = DataSplitConstants(
    hf_split="validation", folder_split="val_small", raw_samples=100000, truncated_samples=100000
)

def get_dataset_constants(dataset_name: str) -> DatasetConstants:
    """Возвращает константы для датасета. Если константы не указаны, возвращает стандартные."""
    return CONSTS.get(dataset_name, DEFAULT_CONSTS)

def main(args: Namespace) -> None:
    """Main: create C4/pile streaming dataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    dataset_constants = get_dataset_constants(args.dataset_path)

    if args.concat_tokens is not None:
        mode = ConcatMode.CONCAT_TOKENS
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        # we will enforce length, so suppress warnings about sequences too long for the model
        tokenizer.model_max_length = int(1e30)
        columns = {"tokens": "bytes"}
    else:
        mode = ConcatMode.NO_CONCAT
        tokenizer = None
        columns = {"text": "str"}

    for split_name in args.splits:
        try:
            split = dataset_constants.splits[split_name]
        except KeyError:
            raise KeyError(f"Constants not defined for split {split_name}.")
        hf_split = split.hf_split
        folder_split = split.folder_split
        expected_num_samples = split.raw_samples
        truncate_num_samples = split.truncated_samples
        # Only generate the splits requested
        if folder_split not in args.splits:
            continue

        # Get samples
        dataset = build_hf_dataset(
            dataset_path=args.dataset_path,
            mode=mode,
            max_length=args.concat_tokens,
            bos_text=args.bos_text,
            eos_text=args.eos_text,
            no_wrap=args.no_wrap,
            tokenizer=tokenizer,
        )
        loader = build_dataloader(dataset=dataset, batch_size=512)
        samples = generate_samples(loader, truncate_num_samples=truncate_num_samples)

        if expected_num_samples is not None:
            denominator = (
                truncate_num_samples
                if truncate_num_samples is not None
                else _est_progress_denominator(
                    total_samples=expected_num_samples,
                    chars_per_sample=dataset_constants.chars_per_sample,
                    chars_per_token=dataset_constants.chars_per_token,
                    mode=mode,
                    max_length=args.concat_tokens,
                )
            )
        else:
            denominator = None

        # Write samples
        print(f"Converting {folder_split} to MDS format...")
        with MDSWriter(
            columns=columns, out=os.path.join(args.out_root, folder_split), compression=args.compression
        ) as out:
            if denominator is not None:
                for sample in tqdm(samples, desc=folder_split, total=denominator):
                    out.write(sample)
            else:
                for sample in tqdm(samples, desc=folder_split):
                    out.write(sample)

if __name__ == "__main__":
    main(parse_args())