GRPC_SOURCES = test_pb2.py test_pb2_grpc.py

all: $(GRPC_SOURCES)

$(GRPC_SOURCES): test.proto
	python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. test.proto

clean:
	rm $(GRPC_SOURCES)
