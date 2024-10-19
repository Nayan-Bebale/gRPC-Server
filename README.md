
# gRPC-Server

This repository contains a Python-based server that utilizes gRPC (Google Remote Procedure Calls) for object detection tasks.

## Features
- Efficient server-client architecture with gRPC.
- Object detection using protocol buffers.
- Scalable and easy to extend.

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Run

1. Define the `.proto` file.
2. Generate Python code:
   ```bash
   python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. object_detection.proto
   ```
3. Run the server:
   ```bash
   python server.py
   ```

## File Structure
- `server.py`: The gRPC server implementation.
- `object_detection.proto`: Protocol buffer definition for object detection.
- `detect.py`: Logic for object detection.

## Usage
Detailed instructions can be found in the `stepsToFollow` file.

## License
This project is licensed under the MIT License.

---
