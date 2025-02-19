# Deploy-ONNX-model
## 1. Download image from NVidia
`docker pull nvcr.io/nvidia/tritonserver:<22.11>-py3-sdk`
## 2. Directory Structure
```
model_repository/
│── classify_cat_dog_onnx/     # Model name
│   ├── 1/                     # Model version
│   │   ├── model.onnx         # ONNX file
│   ├── config.pbtxt           # Triton config file
```

## 3. Launch Triton Inference server
`docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:22.11-py3 tritonserver --model-repository=/models`
Output:
```
I0219 14:15:16.486384 1 server.cc:633]
+---------------+---------+--------+
| Model         | Version | Status |
+---------------+---------+--------+
| densenet_onnx | 1       | READY  |
+---------------+---------+--------+
...
I0219 14:15:16.504751 1 grpc_server.cc:4819] Started GRPCInferenceService at 0.0.0.0:8001
I0219 14:15:16.505880 1 http_server.cc:3477] Started HTTPService at 0.0.0.0:8000
I0219 14:15:16.549719 1 http_server.cc:184] Started Metrics Service at 0.0.0.0:8002
```
## 4. Check Server Status
`curl.exe -v localhost:8000/v2/health/ready`

Output
```
curl -v localhost:8000/v2/health/ready
```
## 5. Using a Triton Client to Query the Server
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
## 6. Triton Performance Analyzer
### 6.1 Start Triton Container
```
 docker pull nvcr.io/nvidia/tritonserver:22.11-py3-sdk
```
