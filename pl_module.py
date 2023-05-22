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

		self.register_loss_params()

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

		if 'mmd' in self.losses:
			ret.update(objectives.compute_mmd(self.model, out))

		if 'wd' in self.losses:
			ret.update(objectives.compute_wd(self.model, out))

		if 'pe' in self.losses:
			ret.update(objectives.compute_pe(self.model, out))

		if 'de' in self.losses:
			ret.update(objectives.compute_de(self.model, out))

		if 'tripe' in self.losses:
			ret.update(objectives.compute_tripe(self.model, out))

		if 'mmpe' in self.losses:
			ret.update(objectives.compute_mmpe(self.model, out))

		if 'vib' in self.losses:
			ret.update(objectives.compute_vib(self.model, out))

		if 'se' in self.losses:
			ret.update(objectives.compute_se(self.model, out))

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

		if "tripe" in self.losses:
			self.model.si_scale = nn.Parameter(self.hparams._config["pe_scale"] * torch.ones(1))
			self.model.si_shift = nn.Parameter(self.hparams._config["pe_shift"] * torch.ones(1))
			self.model.ds_scale = nn.Parameter(self.hparams._config["pe_scale"] * torch.ones(1))
			self.model.ds_shift = nn.Parameter(self.hparams._config["pe_shift"] * torch.ones(1))
			self.model.id_scale = nn.Parameter(self.hparams._config["pe_scale"] * torch.ones(1))
			self.model.id_shift = nn.Parameter(self.hparams._config["pe_shift"] * torch.ones(1))
			self.model.register_parameter('si_scale', self.model.si_scale)
			self.model.register_parameter('si_shift', self.model.si_shift)
			self.model.register_parameter('ds_scale', self.model.ds_scale)
			self.model.register_parameter('ds_shift', self.model.ds_shift)
			self.model.register_parameter('id_scale', self.model.id_scale)
			self.model.register_parameter('id_shift', self.model.id_shift)

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

		if "tripe" in self.losses:
			if phase == "train":
				self.log(f"tripe/{phase}/si_scale", self.model.si_scale)
				self.log(f"tripe/{phase}/si_shift", self.model.si_shift)
				self.log(f"tripe/{phase}/ds_scale", self.model.ds_scale)
				self.log(f"tripe/{phase}/ds_shift", self.model.ds_shift)
				self.log(f"tripe/{phase}/id_scale", self.model.id_scale)
				self.log(f"tripe/{phase}/id_shift", self.model.id_shift)
			tripe_loss = getattr(self, f"{phase}_tripe_loss")(output[f"tripe_loss"])
			i2s = getattr(self, f"{phase}_i2s")(output["i2s"])
			s2i = getattr(self, f"{phase}_s2i")(output["s2i"])
			d2s = getattr(self, f"{phase}_d2s")(output["d2s"])
			s2d = getattr(self, f"{phase}_s2d")(output["s2d"])
			i2d = getattr(self, f"{phase}_i2d")(output["i2d"])
			d2i = getattr(self, f"{phase}_d2i")(output["d2i"])
			i2s_pos = getattr(self, f"{phase}_i2s_pos")(output["i2s_pos"])
			i2s_neg = getattr(self, f"{phase}_i2s_neg")(output["i2s_neg"])
			s2i_pos = getattr(self, f"{phase}_s2i_pos")(output["s2i_pos"])
			s2i_neg = getattr(self, f"{phase}_s2i_neg")(output["s2i_neg"])
			d2s_pos = getattr(self, f"{phase}_d2s_pos")(output["d2s_pos"])
			d2s_neg = getattr(self, f"{phase}_d2s_neg")(output["d2s_neg"])
			s2d_pos = getattr(self, f"{phase}_s2d_pos")(output["s2d_pos"])
			s2d_neg = getattr(self, f"{phase}_s2d_neg")(output["s2d_neg"])
			i2d_pos = getattr(self, f"{phase}_i2d_pos")(output["i2d_pos"])
			i2d_neg = getattr(self, f"{phase}_i2d_neg")(output["i2d_neg"])
			d2i_pos = getattr(self, f"{phase}_d2i_pos")(output["d2i_pos"])
			d2i_neg = getattr(self, f"{phase}_d2i_neg")(output["d2i_neg"])
			vib_loss = getattr(self, f"{phase}_vib_loss")(output["vib_loss"])
			image_vol = getattr(self, f"{phase}_image_volume")(output["image_volume"])
			desc_vol = getattr(self, f"{phase}_description_volume")(output["description_volume"])
			sec_vol = getattr(self, f"{phase}_section_volume")(output["section_volume"])
			self.log(f"tripe/{phase}/tripe_loss", tripe_loss, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"tripe/{phase}/i2s", i2s, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"tripe/{phase}/s2i", s2i, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"tripe/{phase}/d2s", d2s, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"tripe/{phase}/s2d", s2d, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"tripe/{phase}/i2d", i2d, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"tripe/{phase}/d2i", d2i, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"tripe/{phase}/i2s_pos", i2s_pos, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"tripe/{phase}/i2s_neg", i2s_neg, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"tripe/{phase}/s2i_pos", s2i_pos, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"tripe/{phase}/s2i_neg", s2i_neg, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"tripe/{phase}/d2s_pos", d2s_pos, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"tripe/{phase}/d2s_neg", d2s_neg, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"tripe/{phase}/s2d_pos", s2d_pos, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"tripe/{phase}/s2d_neg", s2d_neg, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"tripe/{phase}/i2d_pos", i2d_pos, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"tripe/{phase}/i2d_neg", i2d_neg, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"tripe/{phase}/d2i_pos", d2i_pos, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"tripe/{phase}/d2i_neg", d2i_neg, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"tripe/{phase}/vib_loss", vib_loss, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"tripe/{phase}/image_volume", image_vol, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"tripe/{phase}/description_volume", desc_vol, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"tripe/{phase}/section_volume", sec_vol, batch_size=self.hparams._config["per_gpu_batchsize"])

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

		if "vib" in self.losses:
			vib_loss = getattr(self, f"{phase}_vib_loss")(output["vib_loss"])
			image_vol = getattr(self, f"{phase}_image_volume")(output["image_volume"])
			text_vol = getattr(self, f"{phase}_text_volume")(output["text_volume"])
			self.log(f"vib/{phase}/vib_loss", vib_loss, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"vib/{phase}/image_volume", image_vol, batch_size=self.hparams._config["per_gpu_batchsize"])
			self.log(f"vib/{phase}/text_volume", text_vol, batch_size=self.hparams._config["per_gpu_batchsize"])
