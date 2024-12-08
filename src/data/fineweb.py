from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset
import os


FINEWEB_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/fineweb/")


tknzr = tiktoken.get_encoding("gpt2")


def get_fineweb_data(num_proc=40):
    if not os.path.exists(os.path.join(FINEWEB_DATA_PATH, "train.bin")):
        os.makedirs(FINEWEB_DATA_PATH, exist_ok=True)
        dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", cache_dir="/mloscratch/homes/lcostes/MLerveilleux_project_2/huggingface_cache/datasets")

        split_dataset = dataset["train"].train_test_split(
            test_size=0.0005, seed=2357, shuffle=False
        )
        split_dataset["val"] = split_dataset.pop("test")

        def process(example):
            ids = tknzr.encode_ordinary(
                example["text"]
            )  # encode_ordinary ignores any special tokens
            ids.append(
                tknzr.eot_token
            )  # add the end of text token, e.g. 50256 for gpt2 bpe
            date = int(example["date"][:4])
            #mask_date = torch.zeros((max_date - min_date + 1) // 2)
            #mask_date[:(date - min_date + 1) // 2] = 1
            min_date = 2013
            dates = [(date - min_date + 1) // 2] * len(ids)
            out = {"ids": ids, "len": len(ids), "date": dates}
            return out

        # tokenize the dataset
        tokenized = split_dataset.map(
            process,
            remove_columns=["text"],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )

        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"])
            filename = os.path.join(FINEWEB_DATA_PATH, f"{split}.bin")
            dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            dates = np.memmap(filename, mode="w+", shape=(arr_len,), dtype=np.uint8)
            total_batches = min(1024, len(dset))

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                # Batch together samples for faster write
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                date_batch = np.concatenate(batch["date"])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                dates[idx : idx + len(date_batch)] = date_batch
                idx += len(arr_batch)
            arr.flush()

    train_data = np.memmap(
        os.path.join(FINEWEB_DATA_PATH, "train.bin"), dtype=np.uint16, mode="r"
    )
    train_dates = np.memmap(
        os.path.join(FINEWEB_DATA_PATH, "train_dates.bin"), dtype=np.uint8, mode="r"
    )
    val_data = np.memmap(
        os.path.join(FINEWEB_DATA_PATH, "val.bin"), dtype=np.uint16, mode="r"
    )
    val_dates = np.memmap(
        os.path.join(FINEWEB_DATA_PATH, "val_dates.bin"), dtype=np.uint8, mode="r"
    )
    maxi = max(train_dates.max(), val_dates.max())
    train_dates_one_hot = np.zeros((len(train_data), maxi + 1))
    train_dates_one_hot[:, train_dates] = 1

    val_dates_one_hot = np.zeros((len(val_dates), maxi + 1))
    val_dates_one_hot[:, val_dates] = 1

    return {"train": {"tokens": train_data, "dates": train_dates_one_hot}, "val": {"tokens": val_data, "dates": val_dates_one_hot}}
