import grpc
import test_pb2 as pb2
import test_pb2_grpc as pb2_grpc
import argparse
import numpy as np

CHUNK_SIZE = 1024

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', type=str, default=False)
	return(parser.parse_args())

def gen(audio_file_name):

	config = pb2.RecognitionConfig(
		sample_rate=16000,
		audio_encoding = 'LINEAR16_PCM'
		)

	with open(audio_file_name, 'rb') as f:
		data = f.read(CHUNK_SIZE)
		while data != b'':
			yield pb2.RecognizeRequest(config=config, audio=pb2.RecognitionAudio(content=data))
			data = f.read(CHUNK_SIZE)


class Client:
	def __init__(self):
		self.host = 'localhost'
		self.port = 50051
		self.channel = grpc.insecure_channel('{}:{}'.format(self.host, self.port))
		self.stub = pb2_grpc.TestStub(self.channel)

	def get_stream(self, audio_filename):
		return(self.stub.GetServerStreamingResponse(gen(audio_filename)))

	def get_message(self):
		request = pb2.Message(message="toto")
		return(self.stub.GetServerResponse(request))

if __name__ == '__main__':
	args = get_args()
	client = Client()
	res = client.get_message()
	print(f'{res}')

	# results = client.get_stream(args.path)
	# for res in results:
		# print(f'{res}')
