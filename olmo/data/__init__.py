from pathlib import Path
from typing import Any, Dict, List, Optional, cast
import json
import torch
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import copy
from ..aliases import PathOrStr
from ..config import DataConfig, TrainConfig, EvaluatorConfig
from ..exceptions import OLMoConfigurationError
from ..torch_util import barrier, get_global_rank, get_world_size
from .collator import DataCollator
from .iterable_dataset import IterableDataset
from .memmap_dataset import MemMapDataset
from ..tokenizer import Tokenizer
from datasets import Dataset as HFDataset

__all__ = ["MemMapDataset", "DataCollator", "IterableDataset", "build_eval_dataloader", "build_train_dataloader"]


def build_memmap_dataset(
    train_config: TrainConfig, data_config: DataConfig, include_instance_metadata: bool = True
) -> MemMapDataset:
    paths: List[str]
    metadata: List[Dict[str, Any]] = []
    if data_config.paths:
        if data_config.datasets:
            raise OLMoConfigurationError("DataConfig.paths is mutually exclusive with DataConfig.datasets")
        paths = data_config.paths
        for path in paths:
            metadata.append({"path": str(path)})
    elif data_config.datasets:
        paths = []
        for label in sorted(data_config.datasets.keys()):
            label_paths = data_config.datasets[label]
            paths.extend(label_paths)
            metadata.extend([{"label": label}] * len(label_paths))
    else:
        raise OLMoConfigurationError("One of DataConfig.paths or DataConfig.datasets is required")
    return MemMapDataset(
        *paths,
        chunk_size=train_config.model.max_sequence_length,
        metadata=metadata,
        include_instance_metadata=include_instance_metadata,
        pad_token_id=train_config.model.pad_token_id,
        generate_attention_mask=data_config.generate_attention_mask,
        label_mask_paths=cast(Optional[List[PathOrStr]], data_config.label_mask_paths),
    )


def build_eval_dataloader(
    train_config: TrainConfig,
    eval_config: EvaluatorConfig,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    # dataset = build_memmap_dataset(train_config, data_config, include_instance_metadata=True)
    # dataset = CustomLMDataset(train_config, data_config.dataset_path)
    
    tokenizer = Tokenizer.from_train_config(train_config)  
    with open(eval_config.data.dataset_path, 'r') as f:
        raw_dataset = json.load(f)
    if 'dolma' in eval_config.data.dataset_path or "paragraph" in eval_config.label:
        key_ = 'text' if 'text' in raw_dataset[0] else 'train_context'
        all_data = [x[key_] for x in raw_dataset]
        all_data_tokenized = tokenizer.encode_batch(all_data, add_special_tokens=False)        
        dataset = CustomDataset([{"input_ids": all_data_tokenized[i]} for i in range(len(all_data))])
    elif 'fictional' in eval_config.data.dataset_path:
        if "memorization" in eval_config.label:
            input_key, target_key = "mem_input", "mem_target"
        elif "semantic" in eval_config.label:
            input_key, target_key = "gen_input", "gen_target"
        elif "composition" in eval_config.label:
            input_key, target_key = "hard_gen_input", "hard_gen_target"
            
        probes = [f"{d[input_key][i]} {d[target_key][i]}" for d in raw_dataset for i in range(5)]
        all_probes_tokenized = tokenizer.encode_batch(probes, add_special_tokens=False)  
            
        probes_input = [d[input_key][i] for d in raw_dataset for i in range(5)]
        all_input_tokenized = tokenizer.encode_batch(probes_input, add_special_tokens=False)
        all_labels_tokenized = []
        for input_, probe_ in zip(all_input_tokenized, all_probes_tokenized):
            label_mask = torch.ones_like(torch.tensor(probe_), dtype=torch.bool)
            prompt_length = len(input_)
            label_mask[:prompt_length] = False
            all_labels_tokenized.append(label_mask)

        dataset = CustomDataset([{"input_ids": all_probes_tokenized[i], "label_mask": all_labels_tokenized[i]} 
                                for i in range(len(probes))])

    collator = DataCollator(
        pad_direction=eval_config.data.pad_direction, 
        pad_token_id=train_config.model.pad_token_id)
    if eval_config.data.drop_last:
        # Make sure batch size is small enough.
        samples_per_device = len(dataset) // get_world_size()
        batch_size = min(batch_size, samples_per_device)
        assert batch_size > 0, f"dataset for {eval_config.data.paths} is too small"
    seed = eval_config.data.seed if eval_config.data.seed is not None else train_config.seed
    sampler = DistributedSampler(
        dataset,
        drop_last=False,
        shuffle=False,
        num_replicas=get_world_size(),
        rank=get_global_rank(),
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=eval_config.data.num_workers,
        sampler=sampler,
        pin_memory=eval_config.data.pin_memory,
        prefetch_factor=None if eval_config.data.num_workers == 0 else eval_config.data.prefetch_factor,
        persistent_workers=False if eval_config.data.num_workers == 0 else eval_config.data.persistent_workers,
        timeout=eval_config.data.timeout,
    )


def build_original_train_dataloader(train_config: TrainConfig, world_size: Optional[int] = None) -> DataLoader:
    assert train_config.device_train_batch_size is not None
    collator = DataCollator(
        pad_direction=train_config.data.pad_direction, pad_token_id=train_config.model.pad_token_id
    )
    dataset = build_memmap_dataset(train_config, train_config.data, include_instance_metadata=False)
    work_dir = Path(train_config.save_folder) / "train_data"
    if get_global_rank() == 0:
        if work_dir.is_dir() and not train_config.save_overwrite:
            raise OLMoConfigurationError(
                "train data working directory already exists, use --save_overwrite to overwrite"
            )
        else:
            work_dir.mkdir(exist_ok=True, parents=True)
    barrier()
    seed = train_config.data.seed if train_config.data.seed is not None else train_config.seed
    return DataLoader(
        IterableDataset(
            dataset,  # type: ignore
            train_config.global_train_batch_size,
            seed=seed + (train_config.epoch or 0),
            shuffle=True,
            drop_last=train_config.data.drop_last,
            world_size=world_size,
            work_dir=work_dir,
        ),
        batch_size=train_config.device_train_batch_size,
        drop_last=train_config.data.drop_last,
        collate_fn=collator,
        num_workers=train_config.data.num_workers,
        pin_memory=train_config.data.pin_memory,
        prefetch_factor=None if train_config.data.num_workers == 0 else train_config.data.prefetch_factor,
        persistent_workers=False if train_config.data.num_workers == 0 else train_config.data.persistent_workers,
        timeout=train_config.data.timeout,
    )
    
def build_train_dataloader(train_config: TrainConfig) -> DataLoader:
    assert train_config.device_train_batch_size is not None
    collator = DataCollator(
        pad_direction=train_config.data.pad_direction, pad_token_id=train_config.model.pad_token_id
    )
    # dataset = build_memmap_dataset(train_config, train_config.data, include_instance_metadata=False)
    dataset = HFDataset.load_from_disk(train_config.data.dataset_path)
    work_dir = Path(train_config.save_folder) / "train_data"
    if get_global_rank() == 0:
        if work_dir.is_dir() and not train_config.save_overwrite:
            raise OLMoConfigurationError(
                "train data working directory already exists, use --save_overwrite to overwrite"
            )
        else:
            work_dir.mkdir(exist_ok=True, parents=True)
    barrier()
    seed = train_config.data.seed if train_config.data.seed is not None else train_config.seed
    return DataLoader(
        IterableDataset(
            dataset,  # type: ignore
            train_config.global_train_batch_size,
            seed=seed + (train_config.epoch or 0),
            shuffle=train_config.data_shuffling,
            drop_last=train_config.data.drop_last,
            work_dir=work_dir,
        ),
        batch_size=train_config.device_train_batch_size,
        drop_last=train_config.data.drop_last,
        collate_fn=collator,
        num_workers=train_config.data.num_workers,
        pin_memory=train_config.data.pin_memory,
        prefetch_factor=None if train_config.data.num_workers == 0 else train_config.data.prefetch_factor,
        persistent_workers=False if train_config.data.num_workers == 0 else train_config.data.persistent_workers,
        timeout=train_config.data.timeout,
    )
    
    
def build_custom_dataloader(
    train_config: TrainConfig,
    eval_config: EvaluatorConfig,
) -> DataLoader:
    
    tokenizer = Tokenizer.from_train_config(train_config)  
    # assert train_config.probe_dataset is not None
    with open(eval_config.data.dataset_path, 'r') as f:
        raw_dataset = json.load(f)
    
    all_data = [x['text'] for x in raw_dataset]
    all_data_tokenized = tokenizer.encode_batch(all_data, add_special_tokens=False)
    
    dataset = CustomDataset([{"input_ids": all_data_tokenized[i], "metadata":{"label": eval_config.label}} for i in range(len(all_data))])
    collator = DataCollator(
        pad_direction=train_config.data.pad_direction, pad_token_id=train_config.model.pad_token_id
    )
    seed = train_config.seed
    sampler = DistributedSampler(
        dataset,
        drop_last=False,
        shuffle=False,
        num_replicas=get_world_size(),
        rank=get_global_rank(),
        seed=seed,
    )
    
    return DataLoader(
        dataset,
        # batch_size=train_config.device_train_batch_size,
        batch_size=train_config.device_eval_batch_size,
        collate_fn=collator,
        num_workers=train_config.data.num_workers,
        sampler=sampler,
        pin_memory=train_config.data.pin_memory,
        prefetch_factor=None if train_config.data.num_workers == 0 else train_config.data.prefetch_factor,
        persistent_workers=False if train_config.data.num_workers == 0 else train_config.data.persistent_workers,
        timeout=train_config.data.timeout,
        drop_last=False
    )
    
    
    
class CustomDataset(Dataset):
        def __init__(self, data):
            self.data_d = data
            self.length = len(data)
        
        def __len__(self):
            return self.length

        def __getitem__(self, index):
            return self.data_d[index]
            # return [self.data_d[idx] for idx in index]
        
class CustomLMDataset(Dataset):
        def __init__(self, train_config, dataset_path):
            self.tokenizer = Tokenizer.from_train_config(train_config)  
            print(f"Loading from {dataset_path}")
            with open(dataset_path, 'r') as f:
                self.raw_dataset = json.load(f)
                
            
            # self.dataset = raw_dataset
            self.length = len(self.raw_dataset)
        
        def __len__(self):
            return self.length

        def __getitem__(self, idx):  
            print(idx)
            text = self.raw_dataset[idx]['text']
            # IGNORE_INDEX = -100
            encoding = self.tokenizer(text, max_length=2048, padding="max_length", truncation=True, return_tensors="pt")
            
            input_ids = encoding["input_ids"].squeeze(0)  # Remove batch dimension
            # attention_mask = encoding["attention_mask"].squeeze(0)
            # labels = copy.deepcopy(input_ids)
            # labels[labels == self.tokenizer.pad_token_id] = IGNORE_INDEX
              
            return {"input_ids": input_ids}