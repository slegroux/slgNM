# MDL='/home/syl20/data/en/Models/nemo/nemospeechmodels_1.0.0a5/QuartzNet15x5Base-En.nemo'
MDL='/home/syl20/data/en/Models/nemo/stt_en_quartznet15x5.nemo'
TRAIN_SET='/home/syl20/data/en/webex/webex400_capri/train_data.json'
KENLM_BIN='/home/syl20/Nemo/NeMo/examples/asr/decoders/kenlm/build/bin'

python train_kenlm.py --nemo_model_file ${MDL} \
                          --train_file ${TRAIN_SET} \
                          --kenlm_bin_path ${KENLM_BIN} \
                          --kenlm_model_file klm.mdl \
                          --ngram_length 3