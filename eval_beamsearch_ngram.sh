#!/usr/bin/env bash

# https://ngc.nvidia.com/catalog/collections/nvidia:nemo_asr
# MDL="/home/syl20/Nemo/NeMo/examples/asr/exp/tl_wbx/QuartzNet15x5/2021-02-28_21-33-55/checkpoints/QuartzNet15x5.nemo"
# MDL="/home/syl20/.cache/torch/NeMo/NeMo_1.2.0/QuartzNet15x5Base-En/2b066be39e9294d7100fb176ec817722/QuartzNet15x5Base-En.nemo"
MDL_PATH="/home/syl20/data/en/Models/nemo/"
# MDL="/home/syl20/data/en/Models/nemo/stt_en_quartznet15x5.nemo"
# MDL="/home/syl20/data/en/Models/nemo/stt_en_conformer_ctc_large.nemo"
# MDL="/home/syl20/data/en/Models/nemo/stt_en_citrinet_1024.nemo"
# MDL="/home/syl20/data/en/Models/nemo/stt_en_jasper10x5dr.nemo"
MDL=${MDL_PATH}/nemospeechmodels_1.0.0a5/QuartzNet15x5NR-En.nemo
TEST_DATA=${DATA}/en/webex/webex.tst.json 
# TEST_DATA=${DATA}/en/webex/webex.tiny.json
LM=${DATA}/en/webex/lm/3g-klm.bin
BATCH_SIZE=256

#### RESULTS
# TST:
# MDL      | LM         | alpha | beta | BW  | WER   | Oracle
# Tuned    | greedy     |                    |*25.84 |
#          | 3g-klm.bin | 1.0   | 0.5  | 128 | 22.78 | 20.42
# sttQ15x5 | greedy     |                    |*30.96 | 
#          | 3g-klm.bin | 1.0   | 1.0  | 256 | 27.61 | 25.01
#          | 3g-klm.bin | 1.0   | 1.0  | 128 | 27.86 | 25.66
# Q15x5    | greedy     |                    |*37.63 |
#          | 3g-klm.bin |                    | 33.06 | 30.57
# QNR      | greedy     |                    |*39.3  |
#          | 3g-klm.bin | 1.0   | 1.0  | 128 | 35.01 | 32.97
# J10x5dr  | greedy     |                    |*29.91 |
#          | 3g-klm.bin | 1.0   | 1.0 | 128  | 27.57 | 25.34

python eval_beamsearch_ngram.py --nemo_model_file ${MDL} \
                                     --input_manifest ${TEST_DATA} \
                                     --kenlm_model_file  ${LM} \
                                     --acoustic_batch_size ${BATCH_SIZE} \
                                     --beam_width 128 \
                                     --beam_alpha 1.0 \
                                     --beam_beta 1.0 0.5 \
                                     --preds_output_folder output \
                                     --decoding_mode beamsearch_ngram