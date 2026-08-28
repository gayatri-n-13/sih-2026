# Placeholder for future generated protobuf stubs.

The service currently exposes an HTTP/JSON API that mirrors the gRPC
contract specified in the system prompt:

```
service PreprocessingService {
  rpc Preprocess (PreprocessRequest) returns (JobHandle);
  rpc GetPreprocessStatus (JobHandle) returns (PreprocessResult);
}
```

If the orchestrator requires raw gRPC, the .proto file can be added
here and stubs generated with:

    python -m grpc_tools.protoc -I . preprocessing.proto --python_out=. --grpc_python_out=.
