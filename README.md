# Wikipedia Captioning

Code for the AAAI 2023 paper: "[Show, Interpret and Tell: Entity-aware Contextualised Image Captioning in Wikipedia](https://arxiv.org/abs/2209.10474)"

---
<p align="center">
  <img align="middle" src="./assets/teaser.png" alt="Wikipedia Captioning"/>
</p>

The structure of this repo is as follows:

1. Environment setup 
2. Data preprocessing 
3. How to train/evaluate models

## Set-up environment
This code works in our environment with CUDA 11.4 and NVIDIA A40 GPUs.

First, to install all the dependencies from environment.yml:

```bash
conda env create -f environment.yml
spacy download en_core_web_sm
```

To run experiment with `GPT-2++` and `T5++`, you need to install the `transformers` version added as submodule, which contains with some modifications in the `GPT2` and `T5` classes to enable the models to work with images.
```bash
cd transformers_wc
pip install -e .
```

## WIT dataset
The dataset that we use in the paper is built upon WIT dataset (refer to the original [repository](https://github.com/google-research-datasets/wit) for instructions of downloading the data). While the dataset is multilingual, we primarily focus on its English subset, even though there is no constraints to extend our work to other languages. 

To process and clean data (both image and text) for Wikipedia Captioning task, run:
```bash
python utils/preprocess.py --dset wit/goodnews --data_dir /path/to/original/data/ --save_dir /path/to/save/data/
```
If you want to directly start working on the same dataset as ours, please download the data split from [here](https://cvcuab-my.sharepoint.com/:f:/g/personal/knguyen_cvc_uab_cat/Er_nNnUqoidBk2ETpLO0AI0BVYYC6vAx3xO8fnAL6-LtrA?e=pqxpAy), which is already cleaned and preprocessed.

## Train/Evaluate

To train the models from scratch, run this command:
```bash
python main.py --print-config with cluster dist wit/goodnews data_folder='/path/to/the/data' t5pp/gpt2pp expt_name="t5pp_wit"
```
The data augmentation for training can be set in `modules/data_pool.py`

You can pretrain the models with one of the following objectives `T5/BERT/MNEM` as follows:
```bash
python main.py --print-config with cluster dist wit/goodnews data_folder='/path/to/the/data' t5pp/gpt2pp pt_objective='MNEM/T5/BERT'  expt_name="t5pp_pt_mnem_wit"
```
Then, fine-tune the models on the captioning task for better performance:
```bash
python main.py --print-config with cluster dist wit/goodnews data_folder='/path/to/the/data' t5pp/gpt2pp load_path='/path/to/pretrained/weights' expt_name="t5pp_pt_mnem_wit_ft_goodnews"
```
Note, you need to specify the path to the datafolder `data_folder` as well as experiment name `expt_name`, which indicates the experiment folder created inside `result/` to save the weights. You can also play with different training configurations:

| Argument | Values |
|------|------|
| `expt_name` | Experiment name |
| `dataset` | Dataset name |
| `data_folder` | Path to the dataset directory |
| `transform` | Augmentations applied to training images, specified in `modules/data_pool.py` |
| `text_decoder` | Model to train `gpt2++/t5++` |
| `per_gpu_batchsize` | Batch size per gpu (default: 16) |
| `batchsize` | Batch size for accumulated gradients (default: 256) |
| `distributed` | Distributed training (default: True) |
| `num_gpus` | Number of gpus (default: 2) |
| `num_workers` | Number of workers (default: 8) |
| `test` | Evaluation mode (default: False) |
| `load_path` | Used with `test` = True for evaluation |
| `ckpt_path` | Path to checkpoint file to resume training |
| `result_dir` | Directory to save checkpoints |

Please check the `config.py` for more options and details. 

To evaluate the model on contextualized caption generation, use the following:
```bash
python main.py --print-config with cluster dist wit/goodnews data_folder='/path/to/the/data' t5pp/gpt2pp caption_eval expt_name="t5pp_pt_mnem_wit_ft_goodnews_eval" load_path="/path/to/model/weights"
```

## Model Zoo
TODO

## Citation
TODO
