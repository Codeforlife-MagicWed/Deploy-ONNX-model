# Deploy-ONNX-model
## 1. Down image from NVidia
`docker pull nvcr.io/nvidia/tritonserver:<22.11>-py3-sdk`
## 2. Directory Structure
```
model_repository/
│── classify_cat_dog_onnx/     # Model name
│   ├── 1/                     # Model version
│   │   ├── model.onnx         # ONNX file
│   ├── config.pbtxt           # Triton config file
```

## 2. Launch Triton Inference server
` docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:22.11-py3 tritonserver --model-repository=/models`
## 3. Check Server Status
`curl -v localhost:8000/v2/health/ready`
## 4. Using a Triton Client to Query the Server
```
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient(url="localhost:8000")

import cv2
import numpy as np
img = cv2.imread("img1.jpg")

img = cv2.resize(img, (224, 224))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.transpose((2, 0, 1))

transformed_img = np.expand_dims(img, axis=0).astype(np.float32)
transformed_img = np.array([img], dtype=np.float32)



inputs = httpclient.InferInput("data_0", transformed_img.shape, datatype="FP32")
inputs.set_data_from_numpy(transformed_img, binary_data=True)

outputs = httpclient.InferRequestedOutput("fc6_1", binary_data=True, class_count=1000)

# Querying the server
results = client.infer(model_name="densenet_onnx", inputs=[inputs], outputs=[outputs])
inference_output = results.as_numpy('fc6_1').astype(str)

print(np.squeeze(inference_output)[:5])
```
