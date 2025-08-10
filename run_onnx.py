import sys
import time
import onnxruntime as ort
import numpy as np


# Load model
session = ort.InferenceSession("model.onnx", providers=['CUDAExecutionProvider'])
print(ort.get_device())

# Example: get input name and shape
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
input_dtype = session.get_inputs()[0].type
print(f"Input name: {input_name}, shape: {input_shape}")

# for i, inp in enumerate(session.get_inputs()):
#     print(f"Input {i}: name={inp.name}, shape={inp.shape}, type={inp.type}")

# Prepare dummy input (adapt this to your real input shape)
input_shape = [1, 20, 384, 640]
dummy_input = np.random.randint(0, 2, size=input_shape).astype(np.uint8)
print(f"{dummy_input.shape=}")

for _ in range(10):
    session.run(None, {"input": dummy_input})

num_runs = 10
start = time.time()
for _ in range(num_runs):
    session.run(None, {"input": dummy_input})
end = time.time()

avg_latency = (end - start) / num_runs * 1000  # ms

print(f"Average latency: {avg_latency:.2f} ms over {num_runs} runs")

# # Run inference
# start = time.time()
# outputs = session.run(None, {input_name: dummy_input})
# print(f"Inference time: {(time.time() - start) * 1000:.2f} ms")

# # Print output
# for i, output in enumerate(outputs):
#     print(f"Output {i} shape: {output.shape}")
