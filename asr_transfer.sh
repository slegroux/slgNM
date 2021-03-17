#!/usr/bin/env bash
# (c) Sylvain Le Groux <slegroux@ccrma.stanford.edu>

CONFIG_PATH="conf"
WANDB_NAME="slegroux"
WANDB_PROJECT="transfer2wbx"

# ana
TRAIN="/home/syl20/data/en/an4/train_manifest.json"
VALIDATION="/home/syl20/data/en/an4/test_manifest.json"
EXP="${NM}/exp/an4"

# minilibri
TRAIN="/home/syl20/data/en/librispeech/train-clean-5.json"
VALIDATION="/home/syl20/data/en/librispeech/dev-clean-2.json"
CONF_NAME="minilibri.yaml"
EXP="${NM}/exp/minilibri"

# transfer webex
CONFIG_PATH="conf"
CONF_NAME="quartznet_15x5.yaml"
EXP="${NM}/exp/tl_wbx"
TRAIN="${DATA}/en/webex/webex400_capri/train_data.json"
VALIDATION="${DATA}/en/webex/webex.tst.json"

python -W ignore transfer_learning.py \
    --config-path=${CONFIG_PATH} --config-name=${CONF_NAME} \
    model.train_ds.manifest_filepath=${TRAIN} \
    model.train_ds.batch_size=64 \
    model.train_ds.max_duration=18.0 \
    +model.train_ds.pin_memory=True \
    +model.train_ds.num_workers=110 \
    model.validation_ds.manifest_filepath=${VALIDATION} \
    +model.validation_ds.pin_memory=True \
    +model.validation_ds.num_workers=110 \
    model.validation_ds.batch_size=64 \
    model.optim.lr=0.0015 \
    model.optim.betas=[0.95,0.25] \
    model.optim.weight_decay=0.01 \
    trainer.gpus=8 \
    trainer.max_epochs=250 \
    trainer.val_check_interval=1.0 \
    trainer.log_every_n_steps=10 \
    +trainer.precision=16 \
    +trainer.profiler=True \
    exp_manager.exp_dir=${EXP} \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="slegroux" \
    exp_manager.wandb_logger_kwargs.project="webex400" \
    hydra.run.dir="."

#     model.dropout=0.2 \
#     model.optim.sched.warmup_ratio=0.12 \
#     +trainer.fast_dev_run=True
#     +trainer.amp_level="O1" \
#     trainer.val_check_interval=1.0 \
