############ main + pl_module ############

import os, copy
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS

from config import ex
import objectives
import utils
from modules import build_model
from datamodules import DataModule

from cider.pyciderevalcap.ciderD.ciderD import CiderD

class RLCaptionPlModule(pl.LightningModule):
	def __init__(self, _config):
		super(RLCaptionPlModule, self).__init__()
		self.save_hyperparameters()

		# TODO: 1. init model (write nn.Module with: architecture, forward, sampling, generate)
		self.model = build_model(_config)

		utils.set_metrics(self)
		self.losses = list()
		utils.set_loss(self)

		self.register_loss_params()

		# ===================== Inference ====================== #
		if _config['load_path'] != '' and _config['test']:
			print('Loading pretrained weights for testing...')
			ckpt = torch.load(_config['load_path'], map_location='cpu')
			state_dict = ckpt['state_dict']
			self.load_state_dict(state_dict, strict=False)

	def forward(self, batch):
		ret = dict()

		kwargs = self.prepare_forward()
		out = self.model(batch, self.sc_flag, **kwargs)

		# TODO: 2.implementing traing losses
		if 'lm' in self.losses and not self.sc_flag:
			ret.update({"lm_loss": out["loss"]})

		if 'cider' in self.losses and self.sc_flag:
			loss, reward = self.cider_reward(out, **kwargs)
			ret.update({"cider_loss": loss, "cider_reward": reward})

		if 'clips' in self.losses and self.sc_flag:
			loss, reward = self.clip_reward(out, **kwargs)
			ret.update({"clips_loss": loss, "clips_reward": reward})

		return ret, out

	def training_step(self, batch, batch_idx):
		ret, _ = self(batch)
		self.log_metrics(ret)
		t_loss = sum([ret[f"{l}_loss"] for l in self.losses if f"{l}_loss" in ret])
		return t_loss

	def training_epoch_end(self, outs):
		self.epoch_wrapup()

	def validation_step(self, batch, batch_idx):
		ret, _ = self(batch)
		self.log_metrics(ret)

	def validation_epoch_end(self, outs):
		self.epoch_wrapup()

	def test_step(self, batch, batch_idx):
		# ret, out = self(batch)
		# self.log_metrics(ret)
		# if self.hparams._config['run_caption']:
		# 	return self.model.generate(batch, out, **self.trainer.datamodule.collate_hparams)
		# return
		pass

	def test_epoch_end(self, outs):
		if self.hparams._config['run_caption']:
			utils.caption_wrapup(outs, self.hparams._config)
		if self.hparams._config['run_retrieve']:
			utils.retrieve_wrapup(self)
		self.epoch_wrapup()

	def configure_optimizers(self):
		return utils.set_schedule(self)

	def register_loss_params(self):
		self.sc_flag = False

	def log_metrics(self, output):
		phase = "train" if self.training else "val"

		# TODO: 3. Logging
		for _loss in ["lm"]:
			if _loss in self.losses and not self.sc_flag:
				_loss_val = getattr(self, f"{phase}_{_loss}_loss")(output[f"{_loss}_loss"])
				self.log(f"{_loss}/{phase}/loss", _loss_val, batch_size=self.hparams._config["per_gpu_batchsize"])

		if "cider" in self.losses and self.sc_flag:
			cider_loss = getattr(self, f"{phase}_cider_loss")(output["cider_loss"])
			cider_reward = getattr(self, f"{phase}_cider_reward")(output["cider_reward"])
			self.log(f"cider/{phase}/cider_loss", cider_loss, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"cider/{phase}/cider_reward", cider_reward, batch_size=self.hparams._config["per_gpu_batchsize"])

		if "clips" in self.losses and self.sc_flag:
			clips_loss = getattr(self, f"{phase}_clips_loss")(output["clips_loss"])
			clips_reward = getattr(self, f"{phase}_clips_reward")(output["clips_reward"])
			self.log(f"clips/{phase}/clips_loss", clips_loss, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"clips/{phase}/clips_reward", clips_reward, batch_size=self.hparams._config["per_gpu_batchsize"])

	def epoch_wrapup(self):
		phase = "train" if self.training else "val"
		the_metric = 0

		for loss, v in self.hparams._config["losses"].items():
			if v < 1:
				continue

			value = 0
			if (loss == "lm" and not self.sc_flag) or (loss != "lm" and self.sc_flag):
				loss_val = getattr(self, f"{phase}_{loss}_loss").compute()
				self.log(
					f"{loss}/{phase}/loss_epoch",
					loss_val,
				)
				getattr(self, f"{phase}_{loss}_loss").reset()
				value = loss_val

			the_metric += value

		self.log(f"{phase}/the_metric", the_metric)

		if self.trainer.global_rank == 0:
			import json
			log_dir = self.trainer.logger.log_dir
			if not os.path.exists(os.path.join(log_dir, 'config.json')):
				print(f"Saving config file to {log_dir}/config.json")
				with open(os.path.join(log_dir, 'config.json'), 'w') as f:
					json.dump(self.hparams._config, f, indent=4)

	def prepare_forward(self):
		dec_tokenizer = self.trainer.datamodule.collate_hparams["dec_tokenizer"]
		enc_tokenizer = self.trainer.datamodule.collate_hparams["enc_tokenizer"]
		assert dec_tokenizer is not None and enc_tokenizer is not None

		sample_max_len = self.hparams._config.get("sample_max_len", True)
		use_cache = self.hparams._config.get("use_cache", True)
		# num_beam = self.hparams._config.get("n_beam", 1)
		sample_n = self.hparams._config.get("sample_n", 5)
		# early_stop = self.hparams._config.get("early_stop", True)
		cider_baseline = self.hparams._config.get("cider_baseline", "greedy")
		cider_lambda = self.hparams._config.get("cider_lambda", True)
		clip_lambda = self.hparams._config.get("clip_lambda", True)

		return {
			"dec_tokenizer": dec_tokenizer,
			"enc_tokenizer": enc_tokenizer,
			"sample_max_len": sample_max_len,
			"use_cache": use_cache,
			# "num_beam": num_beam,
			"sample_n": sample_n,
			# "early_stop": early_stop,
			"cider_baseline": cider_baseline,
			"cider_lambda": cider_lambda,
			"clip_lambda": clip_lambda,
		}

	def cider_reward(self, out, **kwargs):
		sample_logprobs = torch.stack(out["sample_logprobs"], dim=1)
		sample_seq, baseline_seq = out["sample_seq"], out["baseline_seq"]

		_bs, _ss = len(out["caption"]), out["sample_seq"].size(0)
		seq_per_img, ml = _ss//_bs, sample_seq.size(1)-1
		assert baseline_seq.size(0) == _bs

		res, gts = OrderedDict(), OrderedDict()
		dec_tokenizer = kwargs["dec_tokenizer"].tokenizer
		sample_res = dec_tokenizer.batch_decode(sample_seq, skip_special_tokens=True)
		baseline_res = dec_tokenizer.batch_decode(baseline_seq, skip_special_tokens=True)
		for i in range(_ss):
			res[i] = sample_res[i]
		for i in range(_bs):
			res[_ss + i] = baseline_res[i]
		for i in range(_bs):
			gts[i] = [out["caption"][i]]

		res_ = [{"image_id": i, "caption": [res[i]]} for i in range(len(res))]
		gts_ = {i: gts[i // seq_per_img] for i in range(_ss)}
		gts_.update({i + _ss: gts[i] for i in range(_bs)})

		_, cider_scores = CiderD_scorer.compute_score(gts_, res_)

		cider_reward = kwargs["cider_lambda"] * cider_scores
		cider_reward = (cider_reward[:_ss].reshape(_bs, seq_per_img) - cider_reward[-_bs:][:, np.newaxis]).reshape(_ss)  # _ss == b*n
		cider_reward = np.repeat(cider_reward[:, np.newaxis], ml, 1)  # (b*n, ml,)

		cider_reward = torch.from_numpy(cider_reward).to(sample_logprobs)
		cider_reward = cider_reward.reshape(-1)
		sample_logprobs = sample_logprobs.gather(2, sample_seq[:, 1:].unsqueeze(2)).squeeze(2)
		sample_logprobs = sample_logprobs.reshape(-1)
		mask = (sample_seq[:, 1:] > dec_tokenizer.pad_token_id).to(sample_logprobs)
		mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1).reshape(-1)

		loss = -sample_logprobs * cider_reward * mask
		# loss = loss.view(_ss, ml).sum(1) / mask.view(_ss, ml).sum(1)
		loss = torch.sum(loss) / torch.sum(mask)
		return loss, cider_reward.mean().item()

	def clip_reward(self, out, **kwargs):
		sample_logprobs = torch.stack(out["sample_logprobs"], dim=1)
		sample_seq = out["sample_seq"]

		_bs, _ss = len(out["image_cls"]), out["sample_seq"].size(0)
		seq_per_img, ml = _ss//_bs, sample_seq.size(1)-1

		dec_tokenizer, enc_tokenizer = kwargs["dec_tokenizer"], kwargs["enc_tokenizer"]

		sample_decoded = dec_tokenizer.tokenizer.batch_decode(sample_seq, skip_special_tokens=True)
		# greedy_decoded = dec_tokenizer.tokenizer.batch_decode(greedy_sample_seq, skip_special_tokens=True)

		with torch.no_grad():
			text_tokens, text_mask = enc_tokenizer.tokenize(sample_decoded + out[kwargs["clip_baseline"]], 77)
			text_tokens, text_mask = text_tokens.to(self.device), text_mask.to(self.device)
			_, text_cls, _ = self.model.text_encoder(text_tokens, mask=text_mask)
			image_cls = torch.cat([utils.repeat_tensor_batch_dim(kwargs["sample_n"], out["image_cls"]), out["image_cls"]], dim=0)

			# normalized features
			text_cls = text_cls / text_cls.norm(dim=-1, keepdim=True)
			image_cls = image_cls / image_cls.norm(dim=-1, keepdim=True)

			clip_scores = torch.matmul(text_cls, image_cls.t()).diag()
			clip_reward = kwargs["clip_lambda"] * clip_scores
			clip_reward = (clip_reward[:_ss].reshape(_bs, seq_per_img) - clip_reward[-_bs:][:, None]).reshape(_ss)  # _ss == b*n
			clip_reward = clip_reward[:, None].expand(-1, ml).reshape(-1)  # (b*n, ml,)

		sample_logprobs = sample_logprobs.gather(2, sample_seq[:, 1:].unsqueeze(2)).squeeze(2)
		sample_logprobs = sample_logprobs.reshape(-1)
		mask = (sample_seq[:, 1:] > dec_tokenizer.tokenizer.pad_token_id).to(sample_logprobs)
		mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1).reshape(-1)

		loss = -sample_logprobs * clip_reward * mask
		# loss = loss.view(_ss, ml).sum(1) / mask.view(_ss, ml).sum(1)
		loss = torch.sum(loss) / torch.sum(mask)
		return loss, clip_reward.mean().item()

class OnEpochStartCallback(pl.Callback):
	def on_train_epoch_start(self, trainer, pl_module):
		_config = pl_module.hparams._config
		epoch = trainer.current_epoch

		# If start self critical training
		if _config["self_critical_after"] != -1 and epoch >= _config["self_critical_after"]:
			pl_module.sc_flag = True
			trainer.callbacks[1].best_model_score = None
			trainer.val_check_interval = 0.05
			init_scorer()

			torch.cuda.empty_cache()
			ll_names = ', '.join([_ll for _ll in pl_module.losses if _ll != 'lm']).strip()
			print(f"====== STARTED Self Critical Sequence Training with objectives: {ll_names}  ======")
			import time; time.sleep(60)  # WORKAROUND: due to gpu memory is not freed after epoch completion

def init_scorer():
	global CiderD_scorer
	CiderD_scorer = CiderD(df="wit-train-words")

class RLDataModule(DataModule):
	def __init__(self, _config, dist=False):
		super().__init__(_config, dist=dist)
		self.epoch_to_start_scst = _config["self_critical_after"]
		self.scst_batchsize = _config["scst_batchsize"]

	def train_dataloader(self) -> TRAIN_DATALOADERS:
		if self.epoch_to_start_scst != -1 and self.trainer.current_epoch >= self.epoch_to_start_scst:
			_bs = self.scst_batchsize
		else:
			_bs = self.batch_size
		loader = DataLoader(
			self.train_dataset,
			batch_size=_bs,
			sampler=self.train_sampler,
			num_workers=self.num_workers,
			pin_memory=True,
			collate_fn=self.collate_fn,
		)
		return loader

@ex.automain
def main(_config):
	_config = copy.deepcopy(_config)
	pl.seed_everything(_config["seed"])

	exp_name = f'{_config["expt_name"]}'

	model = RLCaptionPlModule(_config)
	data_module = RLDataModule(_config, dist=_config['distributed'])

	os.makedirs(_config["result_dir"], exist_ok=True)
	logger = pl_loggers.TensorBoardLogger(
		_config["result_dir"],
		name=f'{exp_name}_seed{_config["seed"]}',
	)

	checkpoint_callback = pl.callbacks.ModelCheckpoint(
		save_top_k=_config["save_top_k"],
		verbose=True,
		monitor="val/the_metric",
		mode="min",
		save_last=True,
	)
	lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
	summary_callback = pl.callbacks.ModelSummary(max_depth=2)
	callbacks = [OnEpochStartCallback(), checkpoint_callback, lr_callback, summary_callback]

	num_gpus = (
		_config["num_gpus"]
		if isinstance(_config["num_gpus"], int)
		else len(_config["num_gpus"])
	)

	grad_steps = _config["batch_size"] // (
			_config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
	)

	trainer = pl.Trainer(
		gpus=_config["num_gpus"],
		num_nodes=_config["num_nodes"],
		precision=_config["precision"],
		accelerator=_config["accelerator"],
		benchmark=True,
		# deterministic=True,
		max_epochs=_config["max_epoch"],
		enable_model_summary=False,
		callbacks=callbacks,
		logger=logger,
		replace_sampler_ddp=False,
		accumulate_grad_batches=grad_steps,
		log_every_n_steps=10,
		flush_logs_every_n_steps=10,
		weights_summary="top",
		fast_dev_run=_config["fast_dev_run"],
		val_check_interval=_config["val_check_interval"],
		num_sanity_val_steps=_config["num_sanity_val_steps"],
		gradient_clip_val=_config["gradient_clip_val"],
		reload_dataloaders_every_n_epochs=1,
		# detect_anomaly=True,
		# track_grad_norm=2
	)

	if not _config["test"]:
		trainer.fit(model, datamodule=data_module, ckpt_path=_config["ckpt_path"])
	else:
		trainer.test(model, datamodule=data_module, ckpt_path=_config["ckpt_path"])
