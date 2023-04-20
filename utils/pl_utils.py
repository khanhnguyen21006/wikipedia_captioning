import torch
import torch.optim as optim
import torch.nn.functional as F
from transformers.optimization import AdamW
from transformers import (
	get_polynomial_decay_schedule_with_warmup,
	get_cosine_schedule_with_warmup,
)
from adamp import AdamP

from .metrics import Accuracy, Scalar


def set_metrics(pl_module):
	for split in ["train", "val"]:
		for k, v in pl_module.hparams._config["losses"].items():
			if v < 1:
				continue
			if k == "lm":
				setattr(pl_module, f"{split}_lm_loss", Scalar())
			elif k == "wd":
				setattr(pl_module, f"{split}_wd_loss", Scalar())
			elif k == "div":
				setattr(pl_module, f"{split}_div_loss", Scalar())
			else:
				setattr(pl_module, f"{split}_{k}_loss", Scalar())


def epoch_wrapup(pl_module):
	phase = "train" if pl_module.training else "val"
	the_metric = 0

	for loss, v in pl_module.hparams._config["losses"].items():
		if v < 1:
			continue

		value = 0

		if loss == "lm":
			lm_loss = getattr(pl_module, f"{phase}_lm_loss").compute()
			pl_module.log(
				f"pe/{phase}/loss_epoch",
				lm_loss,
			)
			getattr(pl_module, f"{phase}_lm_loss").reset()

			value = lm_loss
		elif loss == "wd":
			wd_loss = getattr(pl_module, f"{phase}_wd_loss").compute()
			pl_module.log(
				f"wd/{phase}/loss_epoch",
				wd_loss,
			)
			getattr(pl_module, f"{phase}_wd_loss").reset()

			value = wd_loss
		elif loss == "div":
			div_loss = getattr(pl_module, f"{phase}_div_loss").compute()
			pl_module.log(
				f"div/{phase}/loss_epoch",
				div_loss,
			)
			getattr(pl_module, f"{phase}_div_loss").reset()

			value = div_loss
		else:
			loss = getattr(pl_module, f"{phase}_{loss}_loss").compute()
			pl_module.log(
				f"{loss}/{phase}/loss_epoch",
				loss,
			)
			getattr(pl_module, f"{phase}_{loss}_loss").reset()
			value = loss

		the_metric += value

	pl_module.log(f"{phase}/the_metric", the_metric)


def set_loss(pl_module):
	ll = len(pl_module.hparams._config["losses"].keys())
	if ll == 0:
		raise ValueError(f"Invalid training loss ({ll} is given).")
	pl_module.losses = [
		k for k, v in pl_module.hparams._config["losses"].items() if v >= 1
	]


def set_schedule(pl_module):
	# optim params
	optim_type = pl_module.hparams._config["optimizer"]
	lr = pl_module.hparams._config["learning_rate"]
	wd = pl_module.hparams._config["weight_decay"]

	no_decay = [
		"bias",
		"LayerNorm.bias",
		"LayerNorm.weight",
		"norm.bias",
		"norm.weight",
		"norm1.bias",
		"norm1.weight",
		"norm2.bias",
		"norm2.weight",
	]

	# lr scheduler params
	scheduler = pl_module.hparams._config["lr_scheduler"]
	warmup_steps = pl_module.hparams._config["warmup_steps"]
	decay_power = pl_module.hparams._config["decay_power"]
	end_lr = pl_module.hparams._config["end_lr"]

	optimizer_grouped_parameters = [
		{
			"params": [
				p
				for n, p in pl_module.named_parameters()
				if not any(nd in n for nd in no_decay)
			],
			"weight_decay": wd,
			"lr": lr,
		},
		{
			"params": [
				p
				for n, p in pl_module.named_parameters()
				if any(nd in n for nd in no_decay)
			],
			"weight_decay": 0.0,
			"lr": lr,
		},
	]

	if optim_type == "adamw":
		optimizer = AdamW(
			optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
		)
	elif optim_type == "adamp":
		params = [param for param in pl_module.parameters() if param.requires_grad]
		optimizer = AdamP(
			params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=wd
		)
	elif optim_type == "adam":
		params = [param for param in pl_module.parameters()]
		optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd, amsgrad=True)
	else:
		optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)

	max_steps = (
		len(pl_module.trainer.datamodule.train_dataloader())
		* pl_module.trainer.max_epochs
		// pl_module.trainer.accumulate_grad_batches
	)

	print('Config.max_steps (for step-based scheduler): ', max_steps)
	print('Trainer.max_epochs (for step-based scheduler): ', pl_module.trainer.max_epochs)

	sched = {"interval": "step"}
	if scheduler == 'reduce_lr_on_plateau':
		scheduler = optim.lr_scheduler.ReduceLROnPlateau(
			optimizer, mode='min',
			factor=0.1,
			patience=2,
			verbose=True,
			threshold=1e-4,
			threshold_mode='rel',
			cooldown=0,
			min_lr=end_lr,
			eps=1e-8)
		sched.update({"monitor": "lm/val/loss_epoch", "interval": "epoch"})
	elif scheduler == 'cosine_annealing':
		scheduler = optim.lr_scheduler.CosineAnnealingLR(
			optimizer, T_max=pl_module.trainer.max_epochs)
	elif scheduler == 'with_warmup':
		assert warmup_steps < max_steps, f"warmup steps ({warmup_steps}) has to be smaller than max_steps ({max_steps})"
		if decay_power == "cosine":
			scheduler = get_cosine_schedule_with_warmup(
				optimizer,
				num_warmup_steps=warmup_steps,
				num_training_steps=max_steps,
			)
		else:
			scheduler = get_polynomial_decay_schedule_with_warmup(
				optimizer,
				num_warmup_steps=warmup_steps,
				num_training_steps=max_steps,
				lr_end=end_lr,
				power=decay_power,
			)
	else:
		raise ValueError(f'Invalid scheduler name: {scheduler}')

	sched.update({"scheduler": scheduler})

	return (
		[optimizer],
		[sched],
	)
