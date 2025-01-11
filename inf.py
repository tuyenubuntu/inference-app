import onnxruntime as ort
import numpy as np
from PIL import Image



# Load ONNX model
onnx_model_path = "model/aoiai_detectnet_v1.onnx"
session = ort.InferenceSession(onnx_model_path)

# Lấy thông tin input và output
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
output_names = [output.name for output in session.get_outputs()]

# Hàm xử lý ảnh BMP
def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")  # Chuyển ảnh sang grayscale (1 kênh)
    image = image.resize((960, 128))  # Resize ảnh về [960x128]
    image = np.array(image, dtype=np.float32) / 255.0  # Chuẩn hóa ảnh
    image = np.expand_dims(image, axis=0)  # Add kênh batch
    image = np.expand_dims(image, axis=0)  # Add kênh channel (1 kênh)
    return image

# Chạy dự đoán
def run_inference(image_path):
    input_data = preprocess_image(image_path)
    outputs = session.run(output_names, {input_name: input_data})
    return outputs

# Test
image_path = "e1/09_30-10_IMG_LOG__CAM_2_20231029_162851_736.bmp"
results = run_inference(image_path)
print("Output Shapes:")
for name, output in zip(output_names, results):
    print(f"{name}: {np.array(output).shape}")
