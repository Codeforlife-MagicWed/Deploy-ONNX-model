# Deploy-ONNX-model
## 1. Down image from NVidia
`docker pull nvcr.io/nvidia/tritonserver:<22.11>-py3-sdk`
## 2. Directory Structure
model_repository/
│── classify_cat_dog_onnx/     # Model name
│   ├── 1/                     # Model version
│   │   ├── model.onnx         # ONNX file
│   ├── config.pbtxt           # Triton config file

## 2. Launch Triton Inference server
` docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:22.11-py3 tritonserver --model-repository=/models`
## 3. Using a Triton Client to Query the Server
