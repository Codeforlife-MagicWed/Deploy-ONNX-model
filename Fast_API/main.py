from fastapi import FastAPI, HTTPException, File, UploadFile
import numpy as np
import cv2
import tritonclient.http as httpclient

app = FastAPI(title="Densenet model fast API")

TRITON_SERVER_URL = "localhost:8000"
MODEL_NAME = "densenet_onnx"
MODEL_VERSION = "1"

client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)

@app.get("/")
async def root():
    return {"message": "Triton Inference API is running"}

def infer_with_triton(image: np.ndarray):
    try:
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)

        inputs = httpclient.InferInput("data_0", image.shape, "FP32")
        inputs.set_data_from_numpy(image)

        outputs = httpclient.InferRequestedOutput("fc6_1")
        response = client.infer(MODEL_NAME, model_version=MODEL_VERSION, inputs=[inputs], outputs=[outputs])
        result = response.as_numpy("fc6_1").flatten()  # Chuyển thành 1D array

        top5_indices = np.argsort(result)[-5:][::-1]
        top5_values = result[top5_indices]

        return [{"index": int(idx), "value": float(val)} for idx, val in zip(top5_indices, top5_values)]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling Triton: {str(e)}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        print(f"File contents length: {len(contents)}")
        image = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        predictions = infer_with_triton(image)
        return {"predictions": predictions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))