#!/usr/bin/env bash

CONFIG_PATH="conf"

# # ana
# TRAIN="/home/syl20/data/en/an4/train_manifest.json"
# VALIDATION="/home/syl20/data/en/an4/test_manifest.json"
# CONF_NAME="config.yaml"
# EXP="${NM}/exp/an4"

# minilibri
TRAIN="/home/syl20/data/en/librispeech/train-clean-5.json"
VALIDATION="/home/syl20/data/en/librispeech/dev-clean-2.json"
# CONF_NAME="minilibri.yaml"
CONF_NAME="config.yaml"
EXP="${NM}/exp/minilibri"
MAX_DURATION=18.0
TRAIN_BATCH_SIZE=128
DEV_BATCH_SIZE=128
DROPOUT=0.2
LR=0.015
PROJECT='minilibrispeech'

# # webex
# TRAIN="/home/syl20/data/en/librispeech/train-clean-5.json"
# VALIDATION="/home/syl20/data/en/librispeech/dev-clean-2.json"
# CONF_NAME="minilibri.yaml"
# EXP="${NM}/exp/minilibri"

# # transfer 
# # webex
# TRAIN="/home/syl20/data/en/webex/webex400_capri/train_data.json"
# # VALIDATION="/home/syl20/data/en/webex/webex400_capri/dev_data.json"
# VALIDATION="/home/syl20/data/en/webex/webex.tst.json"
# CONFIG_PATH="conf"
# # CONF_NAME="quartznet_15x5.yaml"
# CONF_NAME="quartznet_5x3.yaml"
# EXP="${NM}/exp/webex"

NUM_EPOCHS=50

python -W ignore speech_to_text.py \
    --config-path=${CONFIG_PATH} --config-name=${CONF_NAME} \
    model.train_ds.manifest_filepath=${TRAIN} \
    model.train_ds.batch_size=${TRAIN_BATCH_SIZE} \
    model.train_ds.max_duration=${MAX_DURATION} \
    +model.train_ds.pin_memory=True \
    model.validation_ds.manifest_filepath=${VALIDATION} \
    +model.validation_ds.pin_memory=True \
    model.validation_ds.batch_size=${DEV_BATCH_SIZE} \
    model.dropout=${DROPOUT} \
    model.optim.lr=${LR} \
    model.optim.betas=[0.95,0.25] \
    model.optim.sched.warmup_steps=500 \
    model.optim.weight_decay=0.001 \
    trainer.gpus=8 \
    trainer.max_epochs=${NUM_EPOCHS} \
    trainer.val_check_interval=1.0 \
    +trainer.precision=16 \
    exp_manager.exp_dir=${EXP} \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="slegroux" \
    exp_manager.wandb_logger_kwargs.project=${PROJECT} \
    hydra.run.dir="."

#     +trainer.fast_dev_run=True
