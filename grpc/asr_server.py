#!/usr/bin/env python

from concurrent import futures
import grpc
import asr_service_pb2
import asr_service_pb2_grpc
from google.protobuf import duration_pb2
from model_api import initialize_model, transcribe_all
from nemo.utils import logging


MDL = 'QuartzNet15x5Base-En'
WAV = '1919-142785-0028.wav'
N_WORKERS = 50

class AsrServiceServicer(asr_service_pb2_grpc.AsrServiceServicer):
    def __init__(self):
        self.model = initialize_model(MDL)
    
    def Recognize(self, request, context):
        # RecognizeRequest as input RecognizeResponse as output
        audio_path = request.audio.uri
        response = asr_service_pb2.RecognizeResponse()
        response.transcript = transcribe_all([audio_path], MDL)[0]
        return response
    
    def StreamingRecognize(self, request_iterator, context):
        request = next(request_iterator)
        for request in request_iterator:
            res = "this is"
            if res:
                response = asr_service_pb2.RecognizeResponse()
                response.



def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=N_WORKERS))
    asr_service_pb2_grpc.add_AsrServiceServicer_to_server(AsrServiceServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    logging.setLevel(logging.ERROR)
    serve()


