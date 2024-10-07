#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29599 -m scripts.train configs/1B/1B_bs128_lr4e4_pubmed_1ep_738k.yaml 