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

OUTPUT:
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

OUTPUT
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
### 6.1 Start the Triton SDK container
```
docker run --rm -it --net host nvcr.io/nvidia/tritonserver:22.11-py3-sdk
```
### 6.2 Run the Perf Analyzer
`perf_analyzer -m densenet_onnx`

#### 6.2.1 Config 1
File config.pbtxt
```
name: "densenet_onnx"
platform: "onnxruntime_onnx"
max_batch_size: 0

input [
  {
    name: "data_0"
    data_type: TYPE_FP32
    dims: [1, 3, 224, 224]
  }
]

output [
  {
    name: "fc6_1"
    data_type: TYPE_FP32
    dims: [1, 1000, 1, 1]
  }
]
```


OUTPUT
```
*** Measurement Settings ***
  Batch size: 1
  Service Kind: Triton
  Using "time_windows" mode for stabilization
  Measurement window: 5000 msec
  Using synchronous calls for inference
  Stabilizing using average latency

Request concurrency: 1
  Client:
    Request count: 295
    Throughput: 16.3814 infer/sec
    Avg latency: 60912 usec (standard deviation 8993 usec)
    p50 latency: 59262 usec
    p90 latency: 73487 usec
    p95 latency: 81824 usec
    p99 latency: 85623 usec
    Avg HTTP time: 60901 usec (send/recv 208 usec + response wait 60693 usec)
  Server:
    Inference count: 296
    Execution count: 296
    Successful request count: 296
    Avg request latency: 59799 usec (overhead 97 usec + queue 95 usec + compute input 25 usec + compute infer 59537 usec + compute output 43 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 16.3814 infer/sec, latency 60912 usec
```
#### 6.2.2 Config 2
File config.pbtxt
```
name: "densenet_onnx"
platform: "onnxruntime_onnx"
max_batch_size: 0

input [
  {
    name: "data_0"
    data_type: TYPE_FP32
    dims: [1, 3, 224, 224]
  }
]

output [
  {
    name: "fc6_1"
    data_type: TYPE_FP32
    dims: [1, 1000, 1, 1]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_CPU
  }
]

optimization {
  execution_accelerators {
    gpu_execution_accelerator : [
      {
        name : "tensorrt"
      }
    ]
  }
}
```

OUTPUT:
```
*** Measurement Settings ***
  Batch size: 1
  Service Kind: Triton
  Using "time_windows" mode for stabilization
  Measurement window: 5000 msec
  Using synchronous calls for inference
  Stabilizing using average latency

Request concurrency: 1
  Client:
    Request count: 489
    Throughput: 27.1281 infer/sec
    Avg latency: 36786 usec (standard deviation 2755 usec)
    p50 latency: 36713 usec
    p90 latency: 40153 usec
    p95 latency: 40869 usec
    p99 latency: 42383 usec
    Avg HTTP time: 36776 usec (send/recv 189 usec + response wait 36587 usec)
  Server:
    Inference count: 489
    Execution count: 489
    Successful request count: 489
    Avg request latency: 35561 usec (overhead 96 usec + queue 115 usec + compute input 23 usec + compute infer 35283 usec + compute output 43 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 27.1281 infer/sec, latency 36786 usec
```

