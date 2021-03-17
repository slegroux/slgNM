#!/usr/bin/env python
# (c) Sylvain Le Groux <slegroux@ccrma.stanford.edu>

import pytorch_lightning as pl
from omegaconf import OmegaConf

from nemo.collections.asr.models import EncDecCTCModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from ruamel.yaml import YAML
from omegaconf import DictConfig
import copy
from IPython import embed

@hydra_runner(config_path="conf", config_name="config")
def main(cfg):    
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    # asr_model = EncDecCTCModel(cfg=cfg.model, trainer=trainer)
    asr_model = EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")
    # TODO: try with optim from config file instead of model itself (check duration is OK) 
#     new_optim = copy.deepcopy(asr_model.cfg.optim)
#     new_optim['lr'] = cfg.model.optim.lr

#     be careful with order of these dependent operations
    asr_model.set_trainer(trainer)
    asr_model.setup_training_data(train_data_config=cfg.model.train_ds)
    asr_model.setup_validation_data(val_data_config=cfg.model.validation_ds)
    new_optim = cfg.model.optim
    asr_model.setup_optimization(optim_config=DictConfig(new_optim))


# update pre-trained model parameters
    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        gpu = 1 if cfg.trainer.gpus != 0 else 0
        trainer = pl.Trainer(
            gpus=gpu,
            precision=cfg.trainer.precision,
            amp_level=cfg.trainer.amp_level,
            amp_backend=cfg.trainer.amp_backend,
        )
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == '__main__':
    main()
