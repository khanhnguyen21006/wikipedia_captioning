import torch
import torch.nn as nn
import pytorch_lightning as pl

from modules import build_model
import objectives
import utils

class PlModule(pl.LightningModule):
	def __init__(self, _config):
		super(PlModule, self).__init__()
		self.save_hyperparameters()

		self.model = build_model(_config)

		# ========= Finetune after training if needed =========== #
		if _config['load_path'] != '' and not _config['test']:
			print('Loading pretrained weights for fine-tuning...')
			ckpt = torch.load(_config['load_path'], map_location='cpu')
			state_dict = ckpt['state_dict']
			self.load_state_dict(state_dict, strict=False)

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
		out = self.model(batch)

		if 'lm' in self.losses:
			decode_kwargs = {
				"dataset" : self.hparams._config['dataset'],
		        "pt_objective": self.hparams._config['pt_objective'],
		        "dec_tokenizer": self.trainer.datamodule.collate_hparams["dec_tokenizer"].tokenizer
		    }
			ret.update(objectives.compute_lm(self.model, out, **decode_kwargs))

		if 'div' in self.losses:
			ret.update(objectives.compute_div(self.model, out))

		if 'mmd' in self.losses:
			ret.update(objectives.compute_mmd(self.model, out))

		if 'wd' in self.losses:
			ret.update(objectives.compute_wd(self.model, out))

		if 'pe' in self.losses:
			ret.update(objectives.compute_pe(self.model, out))

		if 'de' in self.losses:
			ret.update(objectives.compute_de(self.model, out))

		if 'mmpe' in self.losses:
			ret.update(objectives.compute_mmpe(self.model, out))

		if 'vib' in self.losses:
			ret.update(objectives.compute_vib(self.model, out))

		if 'se' in self.losses:
			ret.update(objectives.compute_se(self.model, out))

		if 'ms' in self.losses:
			ret.update(objectives.compute_ms_mod(self.model, out, batch))

		if 'pm' in self.losses:
			ret.update(objectives.compute_pm(self.model, out, batch))

		if 'clip' in self.losses:
			ret.update(objectives.compute_clip(self.model, out))

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
		if self.hparams._config['run_caption']:
			return self.model.generate(batch, out, **self.trainer.datamodule.collate_hparams)
		return
		# pass

	def test_epoch_end(self, outs):
		if self.hparams._config['run_caption']:
			utils.caption_wrapup(outs, self.hparams._config)
		if self.hparams._config['run_retrieve']:
			utils.retrieve_wrapup(self)
		utils.epoch_wrapup(self)

	def configure_optimizers(self):
		return utils.set_schedule(self)

	def register_loss_params(self):
		if "pe" in self.losses or "de" in self.losses:
			self.model.scale = nn.Parameter(self.hparams._config["pe_scale"] * torch.ones(1))
			self.model.shift = nn.Parameter(self.hparams._config["pe_shift"] * torch.ones(1))
			self.model.register_parameter('scale', self.model.scale)
			self.model.register_parameter('shift', self.model.shift)

		if "se" in self.losses:
			if self.hparams._config["se_match"] == 'multi_instance':
				self.model.max_pool = nn.MaxPool2d(self.hparams._config["n_embed"])

	def log_metrics(self, output):
		phase = "train" if self.training else "val"

		for _loss in ["lm", "wd", "div", "mmd"]:
			if _loss in self.losses:
				_loss_val = getattr(self, f"{phase}_{_loss}_loss")(output[f"{_loss}_loss"])
				self.log(f"{_loss}/{phase}/loss", _loss_val, batch_size=self.hparams._config["per_gpu_batchsize"])

		if "pe" in self.losses or "de" in self.losses:
			_loss = "pe" if "pe" in self.losses else "de"
			if phase == "train":
				self.log(f"scale/{phase}/scale", self.model.scale)
				self.log(f"shift/{phase}/shift", self.model.shift)
			loss = getattr(self, f"{phase}_{_loss}_loss")(output[f"{_loss}_loss"])
			i2t = getattr(self, f"{phase}_i2t")(output["i2t"])
			t2i = getattr(self, f"{phase}_t2i")(output["t2i"])
			i2t_pos = getattr(self, f"{phase}_i2t_pos")(output["i2t_pos"])
			i2t_neg = getattr(self, f"{phase}_i2t_neg")(output["i2t_neg"])
			t2i_pos = getattr(self, f"{phase}_t2i_pos")(output["t2i_pos"])
			t2i_neg = getattr(self, f"{phase}_t2i_neg")(output["t2i_neg"])
			self.log(f"{_loss}/{phase}/{_loss}_loss", loss, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"{_loss}/{phase}/i2t", i2t, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"{_loss}/{phase}/t2i", t2i, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"{_loss}/{phase}/i2t_pos", i2t_pos, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"{_loss}/{phase}/i2t_neg", i2t_neg, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"{_loss}/{phase}/t2i_pos", t2i_pos, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"{_loss}/{phase}/t2i_neg", t2i_neg, batch_size=self.hparams._config["per_gpu_batchsize"])

		if "mmpe" in self.losses:
			mmpe_loss = getattr(self, f"{phase}_mmpe_loss")(output["mmpe_loss"])
			i2t = getattr(self, f"{phase}_i2t")(output["i2t"])
			t2i = getattr(self, f"{phase}_t2i")(output["t2i"])
			logsig_l2_loss = getattr(self, f"{phase}_logsig_l2_loss")(output["logsig_l2_loss"])
			mmpe_loss = getattr(self, f"{phase}_mmpe_loss")(output["mmpe_loss"])
			r1_per_batch = getattr(self, f"{phase}_r@1_per_batch")(output["r@1_per_batch"])
			self.log(f"mmpe/{phase}/i2t", i2t, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"mmpe/{phase}/t2i", t2i, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"mmpe/{phase}/mmpe_loss", mmpe_loss, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"mmpe/{phase}/logsig_l2_loss", logsig_l2_loss, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"mmpe/{phase}/r@1_per_batch", r1_per_batch, batch_size=self.hparams._config["per_gpu_batchsize"])

		if "se" in self.losses:
			se_loss = getattr(self, f"{phase}_se_loss")(output["se_loss"])
			i2t = getattr(self, f"{phase}_i2t")(output["i2t"])
			t2i = getattr(self, f"{phase}_t2i")(output["t2i"])
			r1_per_batch = getattr(self, f"{phase}_r@1_per_batch")(output["r@1_per_batch"])
			self.log(f"se/{phase}/i2t", i2t, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"se/{phase}/t2i", t2i, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"se/{phase}/r@1_per_batch", r1_per_batch, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"se/{phase}/se_loss", se_loss, batch_size=self.hparams._config["per_gpu_batchsize"])

		if "clip" in self.losses:
			clip_loss = getattr(self, f"{phase}_clip_loss")(output["clip_loss"])
			i2t = getattr(self, f"{phase}_i2t")(output["i2t"])
			t2i = getattr(self, f"{phase}_t2i")(output["t2i"])
			self.log(f"clip/{phase}/i2t", i2t, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"clip/{phase}/t2i", t2i, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"clip/{phase}/clip_loss", clip_loss, batch_size=self.hparams._config["per_gpu_batchsize"])

		if "ms" in self.losses:
			ms_loss = getattr(self, f"{phase}_ms_loss")(output["ms_loss"])
			for _i in range(self.hparams._config["n_embed"]):
				spacei = getattr(self, f"{phase}_space{_i}")(output[f"space{_i}"])
				self.log(f"ms/{phase}/space{_i}", spacei, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"ms/{phase}/ms_loss", ms_loss, batch_size=self.hparams._config["per_gpu_batchsize"])

		if "vib" in self.losses:
			vib_loss = getattr(self, f"{phase}_vib_loss")(output["vib_loss"])
			image_vol = getattr(self, f"{phase}_image_volume")(output["image_volume"])
			text_vol = getattr(self, f"{phase}_text_volume")(output["text_volume"])
			self.log(f"vib/{phase}/vib_loss", vib_loss, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"vib/{phase}/image_volume", image_vol, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"vib/{phase}/text_volume", text_vol, batch_size=self.hparams._config["per_gpu_batchsize"])
