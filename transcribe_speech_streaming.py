#!/usr/bin/env python

import pyaudio as pa
from frame_asr import FrameASR
import numpy as np
import time 
from omegaconf import OmegaConf
import copy
import nemo.collections.asr as nemo_asr

SAMPLE_RATE= 16000
# duration of signal frame, seconds
FRAME_LEN = 1.0
# number of audio channels (expect mono signal)
CHANNELS = 1

CHUNK_SIZE = int(FRAME_LEN*SAMPLE_RATE)

asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained('QuartzNet15x5Base-En')
cfg = copy.deepcopy(asr_model._cfg)

asr = FrameASR(model_definition = {
				   'sample_rate': SAMPLE_RATE,
				   'AudioToMelSpectrogramPreprocessor': cfg.preprocessor,
				   'JasperEncoder': cfg.encoder,
				   'labels': cfg.decoder.vocabulary
			   },
			   frame_len=FRAME_LEN, frame_overlap=2, 
			   offset=4)

asr.reset()

p = pa.PyAudio()
print('Available audio input devices:')
input_devices = []
for i in range(p.get_device_count()):
	dev = p.get_device_info_by_index(i)
	if dev.get('maxInputChannels'):
		input_devices.append(i)
		print(i, dev.get('name'))

if len(input_devices):
	dev_idx = -2
	while dev_idx not in input_devices:
		print('Please type input device ID:')
		dev_idx = int(input())

	empty_counter = 0

	def callback(in_data, frame_count, time_info, status):
		global empty_counter
		signal = np.frombuffer(in_data, dtype=np.int16)
		text = asr.transcribe(signal)
		if len(text):
			print(text,end='')
			empty_counter = asr.offset
		elif empty_counter > 0:
			empty_counter -= 1
			if empty_counter == 0:
				print(' ',end='')
		return (in_data, pa.paContinue)

	stream = p.open(format=pa.paInt16,
					channels=CHANNELS,
					rate=SAMPLE_RATE,
					input=True,
					input_device_index=dev_idx,
					stream_callback=callback,
					frames_per_buffer=CHUNK_SIZE)

	print('Listening...')

	stream.start_stream()
	
	# Interrupt kernel and then speak for a few more words to exit the pyaudio loop !
	try:
		while stream.is_active():
			time.sleep(0.1)
	finally:        
		stream.stop_stream()
		stream.close()
		p.terminate()

		print()
		print("PyAudio stopped")
	
else:
	print('ERROR: No audio input device found.')