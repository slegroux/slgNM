import grpc
from concurrent import futures
import time
import test_pb2_grpc as pb2_grpc
import test_pb2 as pb2

class Service(pb2_grpc.TestServicer):
    def __init__(self, *args, **kwargs):
        pass

    def GetServerResponse(self, request, context):
        msg = request.message
        res = f" input message: {msg}"
        result = {'message': res, 'received': True}
        return pb2.MessageResponse(**result)

    def GetServerStreamingResponse(self, request_iterator, context):
        request = next(request_iterator)
        for request in request_iterator:
            res = request.audio.content
            yield pb2.RecognizeResponse(audio_bytes=res)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_TestServicer_to_server(Service(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()