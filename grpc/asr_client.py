#!/usr/bin/env python

import logging
import grpc
from numpy.lib.stride_tricks import as_strided
import asr_service_pb2
import asr_service_pb2_grpc
import argparse
import numpy as np

IP = 'localhost'
PORT = '50051'
SR = 16000
# CHUNK_SIZE = 1024
FRAME_LEN = 2
CHUNK_SIZE = int(FRAME_LEN*SR)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--port', type=int, default=50051)
    parser.add_argument('--sr', type=int, default=16000)
    parser.add_argument('--streaming', type=bool, default=False)
    return(parser.parse_args())

def gen(audio_file_name):
	config = asr_service_pb2.RecognitionConfig(
		sample_rate=16000,
		audio_encoding = 'LINEAR16_PCM'
		)

	with open(audio_file_name, 'rb') as f:
		data = f.read(CHUNK_SIZE)
		while data != b'':
			yield asr_service_pb2.RecognizeRequest(
                config=config,
                audio=asr_service_pb2.RecognitionAudio(content=data)
                )
			data = f.read(CHUNK_SIZE)


class AsrServiceClient():
    def __init__(self, ip, port):
        ip_port = f'{ip}:{port}'
        channel = grpc.insecure_channel(ip_port)
        self.stub = asr_service_pb2_grpc.AsrServiceStub(channel)
        self.config = self.audio_config() 

    @staticmethod
    def audio_config(sr=SR, encoding='LINEAR16_PCM'):
        config = asr_service_pb2.RecognitionConfig(
            sample_rate = sr,
            audio_encoding = encoding
        )
        return config

    def get_from_file(self, audio_path):
        request_audio = asr_service_pb2.RecognitionAudio(uri=audio_path)
        request = asr_service_pb2.RecognizeRequest(config=self.config, audio=request_audio)
        return self.stub.Recognize(request)

    def get_from_file_stream(self, audio_path):
        # rand_bytes = np.random.bytes(8)
        # request_audio = asr_service_pb2.RecognitionAudio(content=rand_bytes)
        # request = asr_service_pb2.RecognizeRequest(config=self.config, audio=request_audio)
        return self.stub.StreamingRecognize(gen(audio_path))

    def get_from_mic_stream(self, sr=16000):
        pass


if __name__ == '__main__':
    logging.basicConfig()
    args = get_args()
    client = AsrServiceClient(IP,args.port)
    if args.streaming:
        results = client.get_from_file_stream(args.path)
        for res in results:
            print(res)
    else:
        print(client.get_from_file(args.path))
    