from trism import TritonModel
import numpy as np
import cv2
# Create triton model.
model = TritonModel(
  model="densenet_onnx",     # Model name.
  version=1,            # Model version.
  url="localhost:8001", # Triton Server URL.
  grpc=True             # Use gRPC or Http.
)

# View metadata.
for inp in model.inputs:
  print(f"name: {inp.name}, shape: {inp.shape}, datatype: {inp.dtype}\n")
for out in model.outputs:
  print(f"name: {out.name}, shape: {out.shape}, datatype: {out.dtype}\n")

img = cv2.imread(r"C:\Users\Asus\OneDrive - camann\Desktop\notebook\OJT_Spring2025\Deploy_ONNX_model\Deploy-ONNX-model\model_repository\img1.jpg")
img = cv2.resize(img, (224, 224))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.transpose((2, 0, 1))
transformed_img = np.expand_dims(img, axis=0).astype(np.float32)
# Inference
outputs = model.run(data=[transformed_img])
print(sorted(np.squeeze(outputs['fc6_1']), reverse=True)[:6])
