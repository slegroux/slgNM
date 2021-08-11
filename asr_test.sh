#!/usr/bin/env bash

#### MODELS
# https://ngc.nvidia.com/catalog/collections/nvidia:nemo_asr
# CTC
# MDL="/home/syl20/Nemo/NeMo/examples/asr/exp/tl_wbx/QuartzNet15x5/2021-02-28_21-33-55/checkpoints/QuartzNet15x5.nemo"
# MDL="QuartzNet15x5Base-En"
MDL_PATH="/home/syl20/data/en/Models/nemo"
# MDL="/home/syl20/data/en/Models/nemo/QuartzNet15x5Base-En.nemo"
# MDL="/home/syl20/data/en/Models/nemo/stt_en_quartznet15x5.nemo"
# MDL="stt_en_quartznet15x5"
MDL=${MDL_PATH}/nemospeechmodels_1.0.0a5/QuartzNet15x5NR-En.nemo

#### DATASETS
# TEST_DATA=${DATA}/en/librispeech/dev-clean-2.json
# TEST_DATA=${DATA}/en/webex/webex.tst.json 
TEST_DATA=${DATA}/en/webex/webex.tiny.json

### RESULTS
# MDL   | Tiny | TST
# --------------------
# Tuned | 0.18 | 0.257
# Q15x5 | 0.31 | 0.372

# python speech_to_text_infer.py --asr_model=${MDL} --dataset=${TEST_DATA}

#### LM
# LM_PATH=${DATA}/en/webex/lm/3-gram.train.lower.arpa #(0.258, 16) (0.232, 100)
# LM_PATH=${DATA}/en/webex/lm/3gram_unpruned.arpa #(wer, width): (0.272, 16) (0.23, 100)
# LM_PATH=${DATA}/en/lm/2019_05_10/f_v_r_w_3gram_unpruned.arpa #(0.270, 16) (0.234,100)
# LM_PATH=${DATA}/en/librispeech/lm/lowercase_3-gram.pruned.1e-7.arpa

#### HYPERPARAM
ALPHA=1.0
BETA=1.0
BEAM_WIDTH=128

#### RESULTS
#ALPHA=1.5, BETA=0.9, BW=16
# MDL   | LM                                | TST
# ------------------------------------------------------------------------
# Tuned | 3-gram.train.lower.arpa           | 0.258 (bw16) | 0.232 (bw100)
#       | 3gram_unpruned.arpa               | 0.272        | 0.23
#       | f_v_r_w_3gram_unpruned.arpa       | 0.270        | 0.234
#       | lowercase_3-gram.pruned.1e-7.arpa |

python speech_to_text_infer_lm.py --asr_model=${MDL} --dataset=${TEST_DATA} --lm_path=${LM_PATH} \
    --alpha ${ALPHA} --beta ${BETA} --beam_width ${BEAM_WIDTH}


#### BUFFERED
python speech_to_text_buffered_infer.py --asr_model=${MDL} --test_manifest=${TEST_DATA} --output_path=output