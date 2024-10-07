import argparse
import torch
import json
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, OlmoForCausalLM
from .modeling_olmo_hf import ExpOlmoForCausalLM
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


import os
from pathlib import Path
import requests
from urllib.parse import urlparse

from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset


def main(args):
    if args.model_size == "1B":
        layer_num, mlp_dimension, dim = 16, 8192, 2048
    else:
        layer_num, mlp_dimension, dim = 32, 11008, 4096
        
    dataloader, model, model_path = load_model(args)
    
    
    batch_num = 0
    non_padding_count = 0 
    
    avg_act_abs = torch.zeros((layer_num, mlp_dimension), device=model.device)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(model.device)

            outputs = model(input_ids=input_ids)
            
            logits = outputs.logits
            bs, seq_len, _ = logits.shape
            batch_num += bs
   
            # batch_entropies = []
            for layer_idx, activation in enumerate(outputs.activations):
                # entropy = turn_into_entropy(torch.abs(activation)) # (bs, seq_len)
                # batch_entropies.append(entropy)
                # if args.data_type == "cpt" and 'attention_mask' in batch:
                #     tempmask = entropy[batch['attention_mask'] == 1]
                #     entropy_act[layer_idx] += tempmask.sum().item()
                # else:
                #     entropy_act[layer_idx] += torch.sum(entropy).item()/(entropy.shape[-1])
                
                reshaped_activation_abs = torch.abs(activation).view(-1, mlp_dimension)
                # reshaped_activation = activation.view(-1, mlp_dimension)
                if 'attention_mask' in batch:
                    # flat_attention_mask = batch['attention_mask'].view(-1)
                    # reshaped_activation = reshaped_activation_abs[flat_attention_mask == 1]
                    summed_activation = torch.sum(reshaped_activation_abs, dim=0)
                    avg_act_abs[layer_idx] += summed_activation
                else:    
                    summed_activation = torch.sum(reshaped_activation_abs, dim=0)
                    summed_activation /= activation.shape[1]
                    avg_act_abs[layer_idx] += summed_activation
                    # summed_activation = torch.sum(reshaped_activation, dim=0)
                    # summed_activation /= activation.shape[1]
                    # avg_act[layer_idx] += summed_activation
            # batch_entropies = torch.stack(batch_entropies, dim=-1)
            # all_entropies.append(batch_entropies)
            
            # Clear memory
            # del entropy, reshaped_activation_abs, summed_activation, batch_entropies    
            del input_ids, outputs, logits, reshaped_activation_abs, summed_activation
            torch.cuda.empty_cache() 
    
    length = non_padding_count if non_padding_count != 0 else batch_num    
    avg_act_abs /= length    
    entropy_avg_act_abs = turn_into_entropy(avg_act_abs).tolist() 

    torch.save(avg_act_abs, f"{model_path}/mlp_average_coefficients.pt")
    to_save = {            
        "entropy_avg_act_abs": entropy_avg_act_abs,
    }
        
    print("-"*50)
    print("Knowledge Entropy : ", sum(entropy_avg_act_abs))    
    print("Knowledge Entropy by layer : ", entropy_avg_act_abs)
    print("-"*50)
    write_json_file(f"{model_path}/knowledge_entropy.json", to_save)
        

def custom_collate_fn(batch):
    # Convert the list of 'input_ids' from each sample to a tensor
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    # Stack the tensors into a batch
    input_ids = torch.stack(input_ids)
    return {'input_ids': input_ids}

def turn_into_entropy_softmax(logit):
    prob = torch.softmax(logit, dim=-1)
    log_probabilities = torch.log(prob + 1e-9)
    entropy_act_sparsity = -torch.sum(prob * log_probabilities, dim=-1)
    return entropy_act_sparsity
    
def turn_into_entropy(logit):
    prob = logit / torch.sum(logit, dim=-1, keepdim=True)
    log_probabilities = torch.log(prob + 1e-9)
    entropy_act_sparsity = -torch.sum(prob * log_probabilities, dim=-1)
    return entropy_act_sparsity
 
    
def load_model(args):
            
    step = args.step
    model_size = args.model_size
    model_path = f"checkpoints/pretrained{'_1B' if model_size == '1B' else ''}/{step}-unsharded"
    base_model = "allenai/OLMo-1B-hf"
    
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-hf")
    data_path = "data/dolma_1B/first_1k.json" 
    if not os.path.isfile(data_path):
        print("-"*50, "\nSaving decoded Dolma first batch file.")
        save_dolma()
        
    dataset = read_json_file(data_path)            
    dataset = [d['text'] for d in dataset][:args.data_size]
    print(f"\n Loaded dolma dataset \n length: {len(dataset)} \n example: {dataset[-1]}")

    instances = [d for d in range(len(dataset))]
    subset_dataset = IndexedDataset(dataset, instances, tokenizer=tokenizer) 
    dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False) 
    
    model = ExpOlmoForCausalLM.from_pretrained(base_model, attn_implementation="eager") #, device_map="auto")  
    model = convert_to_hf(model, load_path = model_path, model_size=model_size)
        
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("model_path is ", model_path)
    
    return dataloader, model, model_path

def convert_to_hf(model, load_path=None, ckpt_name=None, model_size="1B"):
    from olmo.checkpoint import load_state_dict
    from olmo.config import TrainConfig
    from olmo.util import clean_opt
    
    pt_name = "model.pt" if ckpt_name is None else f"{ckpt_name}.pt"
    ckpt = load_state_dict(
        load_path, pt_name, local_cache=None, map_location="cpu"
    )    
    
    
    yaml_path = f"configs/official/OLMo-{model_size}.yaml"
    args_list = []
    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args_list])
    cfg.device_eval_batch_size = 1
    
    ## CONVERT TO HF-Format
    olmo_config = cfg.model
    # n_layers = 32
    n_layers = olmo_config.n_layers
    n_heads = olmo_config.n_heads
    dim = olmo_config.d_model
    loaded = ckpt
    dims_per_head = dim // n_heads
    # base = 10000.0
    # inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))


    if olmo_config.n_kv_heads is not None:
        num_key_value_heads = olmo_config.n_kv_heads  # for GQA / MQA
    elif olmo_config.multi_query_attention:  # compatibility with other checkpoints
        num_key_value_heads = 1
    else:
        num_key_value_heads = n_heads

    dims_per_head = dim // n_heads
    state_dict = {}
    for layer_i in range(n_layers):
        filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"
        # Unsharded
        # TODO: Layernorm stuff
        # TODO: multi query attention
        fused_dims = [dim, dims_per_head * num_key_value_heads, dims_per_head * num_key_value_heads]
        q_proj_weight, k_proj_weight, v_proj_weight = torch.split(
            loaded[f"transformer.blocks.{layer_i}.att_proj.weight"], fused_dims, dim=0
        )
        up_proj_weight, gate_proj_weight = torch.chunk(
            loaded[f"transformer.blocks.{layer_i}.ff_proj.weight"], 2, dim=0
        )
        
        state_dict.update({
            f"model.layers.{layer_i}.self_attn.q_proj.weight": q_proj_weight,
            f"model.layers.{layer_i}.self_attn.k_proj.weight": k_proj_weight,
            f"model.layers.{layer_i}.self_attn.v_proj.weight": v_proj_weight,
            f"model.layers.{layer_i}.self_attn.o_proj.weight": loaded[
                f"transformer.blocks.{layer_i}.attn_out.weight"
            ],
            f"model.layers.{layer_i}.mlp.gate_proj.weight": gate_proj_weight,
            f"model.layers.{layer_i}.mlp.down_proj.weight": loaded[f"transformer.blocks.{layer_i}.ff_out.weight"],
            f"model.layers.{layer_i}.mlp.up_proj.weight": up_proj_weight,
        })

            # state_dict[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq

    state_dict.update({
        "model.embed_tokens.weight": loaded["transformer.wte.weight"],
        "lm_head.weight": loaded["transformer.ff_out.weight"]
        if "transformer.ff_out.weight" in loaded
        else loaded["transformer.wte.weight"],
    })
    ### End of Conversion.

    # Load Model.
    model.load_state_dict(state_dict)
    
    return model 

def write_json_file(file_path, res):
    with open(file_path, 'w') as f:
        json.dump(res, f, indent=4)
    print(f"Wrote json file to: {file_path}!")

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        res = json.load(f)
    return res


class IndexedDataset(Dataset):
    def __init__(self, dataset, indices, tokenizer=None, config = None, seq_len=2048):
        self.data = dataset
        self.indices = indices
        self.tokenizer = tokenizer 
        self.seq_len = seq_len
        if config:
            if config.debug_data:
                self.data = self.data[:16]
            
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.tokenizer:
            item_data = self.data[idx]
            encoding = self.tokenizer(item_data, max_length=self.seq_len, padding="max_length", truncation=True, return_tensors="pt")
            input_ids = encoding["input_ids"].squeeze(0) 
            attention_mask = encoding["attention_mask"].squeeze(0) 
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        else:
            actual_idx = self.indices[idx]
            return self.data[actual_idx]

def save_dolma():
            
    epoch = 1
    global_train_examples_seen_this_epoch = 0
    data_order_file_path="data/global_indices/1B/global_indices.npy"
    train_config_path = "configs/official/OLMo-1B.yaml"    
            
    if os.path.isfile(data_order_file_path) and os.path.isfile(train_config_path):

        # Download data-order file if it doesn't exist
        global_indices = np.memmap(data_order_file_path, mode="r+", dtype=np.uint32)
        print(f"\n Loaded dataset \n epoch: {epoch} \n global_train_examples_seen_this_epoch : {global_train_examples_seen_this_epoch}")
        
        # Save first batch of Dolma
        instances = []
        term = 1
        for i in range(2048):
            instances.append(global_indices[global_train_examples_seen_this_epoch+i*term])
            
        cfg = TrainConfig.load(train_config_path)
        dataset = build_memmap_dataset(cfg, cfg.data)
        
        tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-hf")
        to_save = []
        print("\nStart decoding dataset")
        for inst in tqdm(instances,total=2048):
            input_ids = dataset[inst]['input_ids']
            text = tokenizer.batch_decode([input_ids])
            to_save.append({
                "id": int(inst),
                "text": text[0]
            })      
        os.makedirs("data/dolma_1B", exist_ok=True)      
        write_json_file(f"data/dolma_1B/first_1k.json", to_save)
        
    else:
        print("Data order file and config file do not exist. Please run `bash scripts/get_dataorder.sh`")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--data_step", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--data_size", type=int, default=4)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--data_manual_start_num", type=int, default=None)
    parser.add_argument("--data_manual_epoch", type=int, default=None)
    parser.add_argument("--finetuned_path", type=str, default=None)
    parser.add_argument("--model_size", type=str, default="1B")
    parser.add_argument("--bf16", type=bool, default=False)
    parser.add_argument("--temperature", type=bool, default=False)
    parser.add_argument("--save_dolma", type=bool, default=False)
    parser.add_argument("--sparsity", type=bool, default=False)
    parser.add_argument("--attn_initial_temp", type=float, default=None)
    parser.add_argument("--mlp_temp_path", type=str, default=None)
    parser.add_argument("--save_name", type=str, default="")

    args = parser.parse_args()
    main(args)
        