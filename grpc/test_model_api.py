#!/usr/bin/env python
from model_api import initialize_model, transcribe_all

MDL = 'QuartzNet15x5Base-En'
WAV = '1919-142785-0028.wav'

mdl = initialize_model(MDL)
tr = transcribe_all([WAV], MDL)
print(tr[0])