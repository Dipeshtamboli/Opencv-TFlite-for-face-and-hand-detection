import numpy as np
import tensorflow as tf

# Load TFLite model and allocate tensors.
# Here Input is 256*256*3

# interpreter = tf.lite.Interpreter(model_path="hand_landmark_3d.tflite") # output is 21*3
# interpreter = tf.lite.Interpreter(model_path="hand_landmark.tflite") # ouput is 21*2
interpreter = tf.lite.Interpreter(model_path=".models/palm_detection.tflite") # ouput is 21*2
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
print(input_details)
print(output_details)
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data.shape)