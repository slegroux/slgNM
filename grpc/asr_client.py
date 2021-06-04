#!/usr/bin/env python

from audio import AudioStream
import logging
import grpc
from numpy.lib.stride_tricks import as_strided
import asr_service_pb2
import asr_service_pb2_grpc
import argparse
import numpy as np
import queue
import pyaudio

# IP = 'localhost'
# IP='http://dx05.sail.prime.cisco.com/'
IP='172.21.150.75'
PORT = '50051'
SR = 16000
# CHUNK_SIZE = 1024
FRAME_LEN = 2
CHUNK_SIZE = int(FRAME_LEN*SR)


class MicrophoneStream(object):
	"""Opens a recording stream as a generator yielding the audio chunks."""
	def __init__(self, rate, chunk):
		self._rate = rate
		self._chunk = chunk

		# Create a thread-safe buffer of audio data
		self._buff = queue.Queue()
		self.closed = True

	def __enter__(self):
		self._audio_interface = pyaudio.PyAudio()
		self._audio_stream = self._audio_interface.open(
			format=pyaudio.paInt16,
			# The API currently only supports 1-channel (mono) audio
			# https://goo.gl/z757pE
			channels=1, rate=self._rate,
			input=True, frames_per_buffer=self._chunk,
			# Run the audio stream asynchronously to fill the buffer object.
			# This is necessary so that the input device's buffer doesn't
			# overflow while the calling thread makes network requests, etc.
			stream_callback=self._fill_buffer,
		)
		self.closed = False
		return self

	def __exit__(self, type, value, traceback):
		self._audio_stream.stop_stream()
		self._audio_stream.close()
		self.closed = True
		# Signal the generator to terminate so that the client's
		# streaming_recognize method will not block the process termination.
		self._buff.put(None)
		self._audio_interface.terminate()

	def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
		"""Continuously collect data from the audio stream, into the buffer."""
		self._buff.put(in_data)
		return None, pyaudio.paContinue

	def generator(self):
		while not self.closed:
			# Use a blocking get() to ensure there's at least one chunk of
			# data, and stop iteration if the chunk is None, indicating the
			# end of the audio stream.
			chunk = self._buff.get()
			if chunk is None:
				return
			data = [chunk]

			# Now consume whatever other data's still buffered.
			while True:
				try:
					chunk = self._buff.get(block=False)
					if chunk is None:
						return
					data.append(chunk)
				except queue.Empty:
					break

			yield b''.join(data)


def set_recognition_config(sr=16000,enc='LINEAR16_PCM'):
	config = asr_service_pb2.RecognitionConfig(
		sample_rate=sr,
		audio_encoding = enc
		)
	return config

def set_recognition_audio_content(data):
	audio=asr_service_pb2.RecognitionAudio(content=data)
	return(audio)

def set_recognition_audio_uri(uri):
	audio=asr_service_pb2.RecognitionAudio(uri=uri)
	return(audio)

def set_recognize_request(config, audio):
	request = asr_service_pb2.RecognizeRequest(
				config=config,
				audio=audio
				)
	return request

buff = queue.Queue()

def callback(self, in_data, frame_count, time_info, flag):
	buff.put(in_data)
	return None, pyaudio.paContinue

def gen_mic():
	config = set_recognition_config()
	a = AudioStream(mode='r')
	a.sampling_rate = SR
	a.chunk = CHUNK_SIZE
	a.input_device = 0
	a.register_callback(callback)
	chunk = buff.get()
	data = [chunk]
	while True:
		try:
			chunk = buff.get(block=False)
			if chunk is None:
				return
			data.append(chunk)
		except queue.Empty:
			break

	# yield b''.join(data)
		yield asr_service_pb2.RecognizeRequest(
				config=config,
				audio=asr_service_pb2.RecognitionAudio(content= b''.join(data))
				)

def gen_file_stream(audio_file_name):
	config = set_recognition_config()
	with open(audio_file_name, 'rb') as f:
		data = f.read(CHUNK_SIZE)
		while data != b'':
			yield set_recognize_request(config=config, audio=set_recognition_audio_content(data))
			data = f.read(CHUNK_SIZE)
		yield set_recognize_request(config=config, audio=set_recognition_audio_content(data))

def gen_mic_stream():
	config = set_recognition_config()
	mic = MicrophoneStream(SR, 1024)
	with mic as stream:
		for data in stream.generator():
			yield set_recognize_request(config=config, audio=set_recognition_audio_content(data))


class AsrServiceClient():
	def __init__(self, ip, port):
		ip_port = f'{ip}:{port}'
		channel = grpc.insecure_channel(ip_port)
		self.stub = asr_service_pb2_grpc.AsrServiceStub(channel)

	def get_from_file(self, audio_path):
		audio = set_recognition_audio_uri(audio_path)
		config = set_recognition_config()
		request = set_recognize_request(config, audio)
		return self.stub.Recognize(request)

	def get_from_file_stream(self, audio_path):
		return self.stub.StreamingRecognize(gen_file_stream(audio_path))

	def get_from_mic_stream(self):
		return self.stub.StreamingRecognize(gen_mic_stream())


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', type=str, required=False)
	parser.add_argument('--port', type=int, default=50051)
	parser.add_argument('--sr', type=int, default=16000)
	parser.add_argument('--streaming', type=bool, default=False)
	parser.add_argument('--mic', type=bool, default=False)
	return(parser.parse_args())


if __name__ == '__main__':
	logging.basicConfig()
	args = get_args()
	client = AsrServiceClient(IP,args.port)

	if args.streaming and args.path:
		results = client.get_from_file_stream(args.path)
		for res in results:
			print(res)
	elif args.path:
		result = client.get_from_file(args.path)
		print(result)
	elif args.mic:
		results = client.get_from_mic_stream()
		for res in results:
			print(res)
	else:
		print(client.get_from_file(args.path))
	