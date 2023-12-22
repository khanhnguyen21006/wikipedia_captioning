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
This code works in our local environment with CUDA 11.4 and NVIDIA A40 GPUs.

First, install all the dependencies from `environment.yml`:

```bash
conda env create -f environment.yml
spacy download en_core_web_sm
```

To run experiments with `GPT2++` and `T5++`, you need to install the `transformers` version added as submodule, which contains some modifications in the `GPT2` and `T5` classes to enable the models to work with images.
```bash
cd transformers_wc
pip install -e .
```

## WIT Dataset
The dataset that we use in the paper is built upon WIT dataset (refer to the original [repository](https://github.com/google-research-datasets/wit) for instructions of downloading the data). While the dataset is multilingual, we primarily focus on its English subset, even though there is no constraints to extend our work to other languages.

To process and clean data (both image and text) for Wikipedia Captioning task, run:
```bash
python utils/preprocess.py --task prep_wit --data_dir /path/to/original/data/ --save_dir /path/to/save/data/
```
If you want to directly start working on the same dataset as ours, please download the data split from [here](https://cvcuab-my.sharepoint.com/:f:/g/personal/knguyen_cvc_uab_cat/Er_nNnUqoidBk2ETpLO0AI0BVYYC6vAx3xO8fnAL6-LtrA?e=pqxpAy), which is already cleaned and preprocessed.

The total size of our dataset constructed from WIT is `~386GB` . Due to the per-file size limits in `onedrive`, we have to split the `train_IMAGE_wit.hdf5` into 3 partitions `0`, `1` and `2` as in the `wit` folder. Thus, you need to do an extra step to join them together (then you can remove the partitions):
```bash
python utils/preprocess.py --task concat_hdf5 --save_dir /path/to/save/data/
```
Now, we are ready to go!

## Train/Evaluate

To train the models `GPT2++/T5++` from scratch, run this command:
```bash
python main.py --print-config with cluster dist wit data_folder='/path/to/the/data' t5pp expt_name="t5pp_wit"
```
The data augmentation for training can be set in `modules/data_pool.py`

You can pre-train the models with one of the following objectives `T5/BERT/MNEM` as follows:
```bash
python main.py --print-config with cluster dist wit data_folder='/path/to/the/data' t5pp pt_objective='MNEM' expt_name="t5pp_pt_mnem_wit"
```
Then, fine-tune the models on the captioning task to see better performance:
```bash
python main.py --print-config with cluster dist wit data_folder='/path/to/the/data' t5pp load_path='/path/to/pretrained/weights' expt_name="t5pp_pt_mnem_wit_ft_wit"
```

Note, you need to specify the path to the datafolder `data_folder` as well as experiment name `expt_name`, which indicates the experiment folder created in `result/` (by default) to save the weights. You can also play with different training configurations:

| Argument | Values |
|------|------|
| `expt_name` | Experiment name |
| `dataset` | Dataset name `wit/goodnews` |
| `data_folder` | Path to the dataset directory |
| `transform` | Augmentations applied to training images, specified in `modules/data_pool.py` |
| `text_decoder` | Model to train `gpt2++/t5++` |
| `per_gpu_batchsize` | Batch size per gpu (default: `16`) |
| `batchsize` | Batch size for accumulated gradients (default: `256`) |
| `distributed` | Distributed training (default: `True`) |
| `num_gpus` | Number of gpus (default: `2`) |
| `num_workers` | Number of workers (default: `8`) |
| `test` | Evaluation mode (default: `False`) |
| `load_path` | Used with `test=True` for evaluation |
| `ckpt_path` | Path to checkpoint file to resume training |
| `result_dir` | Directory to save checkpoints (default: `result/`)|

Please check the `config.py` and [pytorch-lighting](https://pytorch-lightning.readthedocs.io/en/1.5.10/common/trainer.html#trainer-flags) for more options and details. 

To evaluate the model on contextualized caption generation, use the following:
```bash
python main.py --print-config with cluster dist wit data_folder='/path/to/the/data' t5pp caption_eval expt_name="expt_name_eval" load_path="/path/to/model/weights"
```

## Model Zoo
We provide [here](https://cvcuab-my.sharepoint.com/:f:/g/personal/knguyen_cvc_uab_cat/Eh1TL57nukdDmbd8PAGIdzUBgx5XuVFfICWoJbdNJL8J2w) the resulting weights of `T5++` variants (`T5+resnet152`) trained on different settings. Unless specified aside, models are trained with the language modelling objective.
| Pre-train | Fine-tune | Weights |
|------|------|------|
| `WIT` |  | [link](https://cvcuab-my.sharepoint.com/:f:/g/personal/knguyen_cvc_uab_cat/EtULUlYp8vZHve-7r7HvyLsBZv88xPAGV_cNTIYpGuJyJw?e=gtFgdn) |
| `WIT+MNEM` |  | [link](https://cvcuab-my.sharepoint.com/:f:/g/personal/knguyen_cvc_uab_cat/EgwpXGudCfFPmBQNjTmqLPsB_t3Nq3ihSlqzHHbNhqa5fA?e=D7Cbqk) |
| `WIT+T5` |  | [link]() |
| `WIT+BERT` |  | [link]() |
| `WIT+MNEM` | `WIT` | [link](https://cvcuab-my.sharepoint.com/:f:/g/personal/knguyen_cvc_uab_cat/Ejj2AluQC-tPjqHiF_g-TNwBoJznuGPiGpRhE-uDZjvuyw?e=2TXF1l) |
| `WIT` | `GoodNews` | [link](https://cvcuab-my.sharepoint.com/:f:/g/personal/knguyen_cvc_uab_cat/EhZKCPND7tFOjceC9IvrsyoBWR-_2Fih25RxGlhs5xgZNg?e=AxSDzF) |
| `WIT` | `GoodNews+MNEM` | [link](https://cvcuab-my.sharepoint.com/:f:/g/personal/knguyen_cvc_uab_cat/Ep2W_2JeY2hNi19CvkkiPhcBFGgyFM8QlYt1bc0Uwv9pdw?e=fXuduh) |

## Conclusion
Thank you for your interest and sorry for the bugs!
