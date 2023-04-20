import os

import torch
import torch.nn as nn
import pytorch_lightning as pl

from models import Model
import objectives
import utils


class PlayGround(pl.LightningModule):
	def __init__(self, _config):
		super(PlayGround, self).__init__()
		self.save_hyperparameters()

		self.model = Model(_config)

		# ========= Finetune after training if needed =========== #
		if _config['load_path'] != '' and not _config['test']:
			print('Loading pretrained weights for fine-tuning...')
			ckpt = torch.load(_config['load_path'], map_location='cpu')
			state_dict = ckpt['state_dict']
			self.load_state_dict(state_dict, strict=False)

		utils.set_metrics(self)
		self.losses = list()
		utils.set_loss(self)

		# ===================== Inference ====================== #
		if _config['load_path'] != '' and _config['test']:
			print('Loading pretrained weights for testing...')
			ckpt = torch.load(_config['load_path'], map_location='cpu')
			state_dict = ckpt['state_dict']
			self.load_state_dict(state_dict, strict=False)

	def forward(self, batch):
		ret = dict()
		out = self.model.encode(batch)

		if 'lm' in self.losses:
			ret.update(objectives.compute_lm(self.model, out))

		if 'div' in self.losses:
			ret.update(objectives.compute_div(self.model, out))

		if 'wd' in self.losses:
			ret.update(objectives.compute_wd(self.model, out))

		return ret, out

	def training_step(self, batch, batch_idx):
		ret, _ = self(batch)
		self.log_metrics(ret)
		t_loss = sum([ret[f'{l}_loss'] for l in self.losses if f'{l}_loss' in ret])
		return t_loss

	def training_epoch_end(self, outs):
		utils.epoch_wrapup(self)

	def validation_step(self, batch, batch_idx):
		ret, _ = self(batch)
		self.log_metrics(ret)

	def validation_epoch_end(self, outs):
		utils.epoch_wrapup(self)

	def test_step(self, batch, batch_idx):
		ret, out = self(batch)
		self.log_metrics(ret)
		return self.model.generate(batch, out, tokenizer=self.trainer.datamodule.dec_tokenizer)

	def test_epoch_end(self, outs):
		utils.utility_wrapup(outs, self.hparams._config)
		utils.epoch_wrapup(self)

	def configure_optimizers(self):
		return utils.set_schedule(self)

	def log_metrics(self, output):
		phase = "train" if self.training else "val"

		if "lm" in self.losses:
			lm_loss = getattr(self, f"{phase}_lm_loss")(output["lm_loss"])
			self.log(f"lm/{phase}/loss", lm_loss, batch_size=self.hparams._config["per_gpu_batchsize"])

		if "wd" in self.losses:
			wd_loss = getattr(self, f"{phase}_wd_loss")(output["wd_loss"])
			self.log(f"wd/{phase}/loss", wd_loss, batch_size=self.hparams._config["per_gpu_batchsize"])

		if "div" in self.losses:
			div_loss = getattr(self, f"{phase}_div_loss")(output["div_loss"])
			self.log(f"div/{phase}/loss", div_loss, batch_size=self.hparams._config["per_gpu_batchsize"])
