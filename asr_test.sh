#!/usr/bin/env bash


MDL="${NM}/examples/asr/exp/tl_wbx/QuartzNet15x5/2021-02-28_21-33-55/checkpoints/QuartzNet15x5.nemo"
# MDL="QuartzNet15x5Base-En"
# MDL="QuartzNet15x5NR-En"
# MDL="Jasper10x5Dr-En"

# TEST_DATA=${DATA}/en/librispeech/dev-clean-2.json
TEST_DATA=${DATA}/en/webex/webex.tst.json
# TEST_DATA=${DATA}/en/webex/webex.tiny.json

python speech_to_text_infer.py --asr_model=${MDL} --dataset=${TEST_DATA}