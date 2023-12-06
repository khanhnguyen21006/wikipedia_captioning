import os
import copy
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.utilities.model_summary import ModelSummary

from config import ex

from datamodules import DataModule
from pl_module import PlModule


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    exp_name = f'{_config["expt_name"]}'

    model = PlModule(_config)
    data_module = DataModule(_config, dist=_config['distributed'])

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
    callbacks = [checkpoint_callback, lr_callback, summary_callback]

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
        deterministic=True,
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
    )

    if not _config["test"]:
        trainer.fit(model, datamodule=data_module, ckpt_path=_config["ckpt_path"])
    else:
        trainer.test(model, datamodule=data_module, ckpt_path=_config["ckpt_path"])
