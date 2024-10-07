# Knowledge Entropy Decay during Language Model Pretraining Hinders New Knowledge Acquisition

This repository contains the implementation of [**Knowledge Entropy Decay during Language Model Pretraining Hinders New Knowledge Acquisition**](https://arxiv.org/abs/2410.01380). 


![main_figure.pdf](assets/main_figure.pdf)


The code is based on the [OLMo](https://github.com/allenai/OLMo) project, with modifications to support knowledge injection during training and additional evaluations.
Knowledge injection code is based on the [factual-knowledge-acquisition](https://github.com/kaistAI/factual-knowledge-acquisition.git) project, with minor modifications for data split of 'Paraphrase or Once' and evaluations.


## Key Differences from Original OLMo Repository

1. Modified `olmo/train.py` to:
   - Apply knowledge injection during training
2. Modified `olmo/checkpoint.py` to load model checkpoint from resuscitation method
3. Added calculation of Knowledge Entropy for pretrained model in `analysis/` folder
4. Added modification of model checkpoints for resuscitation method in `analysis/` folder

## Key Differences from Original factual-knowledge-acquisition Repository
1. Augmented the number of Paraphrase data with the method introduced in the original paper [How Do Large Language Models Acquire Factual Knowledge During Pretraining?](https://arxiv.org/abs/2406.11813) using GPT4 
    - Original Fictional Knowledge dataset can be found at: https://huggingface.co/datasets/kaist-ai/fictional-knowledge
2. Modified evaluation code


## Overview

This repository is for reproducing calculation of Knowledge Entropy and continually training the intermediate OLMo checkpoints or modified checkpoint with resuscitation method.There are three sections in this repo:

- [Inference and evaluation](#inference-and-evaluation) of Janus models on Multifaceted Bench
- [Training main Janus models](#train-main-models) using [axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)
- [Training Janus-RM-7B](#train-reward-model) using [OpenRLHF](https://github.com/OpenLLMAI/OpenRLHF)


## Installation

1. Create a conda environment with python>=3.9.
```
conda create -n knowledge-entropy python=3.11 -y
conda activate knowledge-entropy
```

2. Install packages.
```
pip install -e .
```

3. Download the intermediate OLMo checkpoint. 
Example of downloading from the official repository is presented in `scripts/get_model.sh`.
First you have to figure out the link to the proper checkpoint in the [official link](https://github.com/allenai/OLMo/blob/main/checkpoints/official/OLMo-1B.csv)
Below is example command for downloading the model whose pretraining step is 738020.

```
bash scripts/get_model.sh https://olmo-checkpoints.org/ai2-llm/olmo-small/oupb6jak/step738020-unsharded/
```


## Knowledge Entropy
1. Download official training order of Dolma from [official link](https://olmo-checkpoints.org/ai2-llm/olmo-small/46zc5fly/train_data/global_indices.npy)

```
bash scripts/get_dataorder.sh
```

2. Run Knowledge entropy calculation command, where step, data_size and batch size should be chosed appropriately.
```
python -m analysis.entropy --step 738020 --data_size 2048 --batch_size 4
```

## Training
Train OLMo model with modified config.
You can find exmaplary config at `configs/1B/1B_bs128_lr4e4_pubmed_1ep_738k.yaml`.


```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29599 -m scripts.train configs/1B/1B_bs128_lr4e4_pubmed_1ep_738k.yaml 
```

## Resuscitation
1. Save modified model chekpoint.
Run modifying and saving model checkpoint for resuscitation method.
Below is examplary command for changing with resuscitation ratio of 50% and amplifying factor of 2. 

```
python -m analysis.change_parameters --step 738020--resuscitation_ratio 0.5 --amplifying_factor 2
```

2. Run training command with modified config. 
The name of the newly saved model should be specified at model.resuscitation.
You can find exmaplary config at `configs/resuscitation/1B_bs128_lr4e4_pubmed_1ep_738k_resuscitation.yaml`.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29599 -m scripts.train configs/resuscitation/1B_bs128_lr4e4_pubmed_1ep_738k_resuscitation.yaml
```


## Citation
If you find our work helpful for your work, please consider citing our paper, original OLMo paper and factual-knowledge-acquisition paper
```bibtex
@article{lee2024aligning,
  title={Aligning to Thousands of Preferences via System Message Generalization},
  author={Lee, Seongyun and Park, Sue Hyun and Kim, Seungone and Seo, Minjoon},
  journal={arXiv preprint arXiv:2405.17977},
  year={2024}
}
@article{chang2024large,
  title={How Do Large Language Models Acquire Factual Knowledge During Pretraining?},
  author={Chang, Hoyeon and Park, Jinho and Ye, Seonghyeon and Yang, Sohee and Seo, Youngkyung and Chang, Du-Seong and Seo, Minjoon},
  journal={arXiv preprint arXiv:2406.11813},
  year={2024}
}

@article{OLMo,
  title={OLMo: Accelerating the Science of Language Models},
  author={Dirk Groeneveld and Iz Beltagy and Pete Walsh and Akshita Bhagia and Rodney Kinney and Oyvind Tafjord and A. Jha and Hamish Ivison and Ian Magnusson and Yizhong Wang and Shane Arora and David Atkinson and Russell Authur and Khyathi Raghavi Chandu and Arman Cohan and Jennifer Dumas and Yanai Elazar and Yuling Gu and Jack Hessel and Tushar Khot and William Merrill and Jacob Daniel Morrison and Niklas Muennighoff and Aakanksha Naik and Crystal Nam and Matthew E. Peters and Valentina Pyatkin and Abhilasha Ravichander and Dustin Schwenk and Saurabh Shah and Will Smith and Emma Strubell and Nishant Subramani and Mitchell Wortsman and Pradeep Dasigi and Nathan Lambert and Kyle Richardson and Luke Zettlemoyer and Jesse Dodge and Kyle Lo and Luca Soldaini and Noah A. Smith and Hanna Hajishirzi},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:267365485},
  journal={arXiv preprint},
}
```
