#!/usr/bin/env bash

# MDL='/home/syl20/Nemo/NeMo/examples/asr/libri/QuartzNet15x5/2021-02-04_06-24-35/checkpoints/QuartzNet15x5.nemo'
MDL="QuartzNet15x5Base-En"
# MDL="QuartzNet15x5NR-En"
# MDL="Jasper10x5Dr-En"


# TEST_DATA=${DATA}/en/webex/webex.tst.json
# LM_PATH=${DATA}/en/mls_lm_english/3-gram_lm.arpa
# LM_PATH=${DATA}/en/lm/2019_05_10/f_v_r_w_3gram_unpruned.arpa
# LM_PATH=data/webex/3-gram.train.lower.arpa
TEST_DATA=${DATA}/en/librispeech/dev-clean-2.json
LM_PATH=${DATA}/en/librispeech/lm/lowercase_3-gram.pruned.1e-7.arpa
ALPHA=2.0
BETA=1.5
BEAM_WIDTH=10

python speech_to_text_infer_lm.py --asr_model=${MDL} --dataset=${TEST_DATA} --lm_path=${LM_PATH} \
    --alpha ${ALPHA} --beta ${BETA} --beam_width ${BEAM_WIDTH}