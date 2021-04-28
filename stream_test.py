#!/usr/bin/env python

import pyaudio as pa
from frame_asr import FrameASR, preprocessor_normalization
import numpy as np
import time 
from omegaconf import OmegaConf
import copy
import nemo.collections.asr as nemo_asr
import numpy as np
import wave
from IPython import embed

SAMPLE_RATE= 16000
# duration of signal frame, seconds
FRAME_LEN = 1.0
# number of audio channels (expect mono signal)
CHANNELS = 1
CHUNK_SIZE = int(FRAME_LEN*SAMPLE_RATE)

wf = wave.open('data/test.wav', 'rb')

asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained('QuartzNet15x5Base-En')
cfg = copy.deepcopy(asr_model._cfg)
cfg = preprocessor_normalization(cfg)
asr_model.preprocessor = asr_model.from_config_dict(cfg.preprocessor)
# inference mode
asr_model.eval()
asr_model = asr_model.to(asr_model.device)

asr = FrameASR(asr_model, cfg,
			   frame_len=FRAME_LEN, frame_overlap=2, 
			   offset=4)

asr.reset()
# res = []
# data = wf.readframes(CHUNK_SIZE)
# res += data
# while data != b'':
# 	data = wf.readframes(CHUNK_SIZE)
# 	res+=data
# text = asr.transcribe(data)
# print(text)

data = wf.readframes(CHUNK_SIZE)
while data != b'':
	data = wf.readframes(CHUNK_SIZE)
	signal = np.frombuffer(data, dtype=np.int16)
	text = asr.transcribe(signal)

	if len(text):
		print(text,end='')

