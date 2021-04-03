# Tiny ASR grpc client/server example
## Install
### grpc
```bash
conda install grpcio grpcio-tools
```
### pytorch and asr models
c.f. nm

### protobuf
- define protobuf interface file (asr_service.proto)
- compile proto to generate client/server template code
```bash
mv grpc
make
```
## Usage
### server
```bash
python asr_server.py
```
### client
```bash
python asr_client.py --path 1919-142785-0028.wav
```
## Contact 
<slegroux@ccrma.stanford.edu>