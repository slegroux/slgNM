GRPC_SOURCES = asr_service_pb2.py asr_service_pb2_grpc.py

all: $(GRPC_SOURCES)

$(GRPC_SOURCES): asr_service.proto
	python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. asr_service.proto

clean:
	rm $(GRPC_SOURCES)
