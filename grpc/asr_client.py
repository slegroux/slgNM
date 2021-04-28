#!/usr/bin/env python
import logging
import grpc
import asr_service_pb2
import asr_service_pb2_grpc
import argparse
IP='localhost'
PORT='50051'
SR=16000

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--port', type=int, default=50051)
    parser.add_argument('--sr', type=int, default=16000)
    return(parser.parse_args())

class AsrServiceClient():
    def __init__(self, ip, port):
        ip_port = f'{ip}:{port}'
        channel = grpc.insecure_channel(ip_port)
        self.stub = asr_service_pb2_grpc.AsrServiceStub(channel)

    def get_transcript(self, audio_path, sr=16000):
        request_config = asr_service_pb2.RecognitionConfig(sample_rate=sr)
        request_config.audio_encoding = asr_service_pb2.RecognitionConfig.LINEAR16_PCM
        request_audio = asr_service_pb2.RecognitionAudio(uri=audio_path)
        request = asr_service_pb2.RecognizeRequest(config=request_config, audio=request_audio)
        return self.stub.Recognize(request).transcript
    
if __name__ == '__main__':
    logging.basicConfig()
    args = get_args()
    client = AsrServiceClient(IP,args.port)
    print(client.get_transcript(args.path, sr=args.sr))
