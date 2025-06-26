import sys
import time
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession("SAST.onnx", providers=['CUDAExecutionProvider'])

# Example: get input name and shape
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f"Input name: {input_name}, shape: {input_shape}")

# Prepare dummy input (adapt this to your real input shape)
dummy_input = np.random.rand(1, 20, 384, 640).astype(np.uint8)
print(dummy_input.shape)

# Run inference
start = time.time()
outputs = session.run(None, {input_name: dummy_input})
print(f"Inference time: {(time.time() - start) * 1000:.2f} ms")

# Print output
for i, output in enumerate(outputs):
    print(f"Output {i} shape: {output.shape}")
