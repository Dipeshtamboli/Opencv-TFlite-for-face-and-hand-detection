import cv2
import tkinter
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

# import matplotlib.image as mpimg
img = "/home/dipesh/Hackathon-where-is-my-sanitizer/images/hand.jpg"
# plt_img = mpimg.imread(img)

frame = cv2.imread(img)
print(frame.shape)
frame = cv2.resize(frame, (256,256), interpolation = cv2.INTER_AREA)
print(frame.shape)
frame = np.expand_dims(frame,0)
# print(frame.shape)
# print(type(frame))
frame = np.array(frame, dtype = np.float32)
# print(type(frame))
# exit()

import tensorflow as tf
# Here Input is 256*256*3
# interpreter = tf.lite.Interpreter(model_path="hand_landmark_3d.tflite") # output is 21*3
# interpreter = tf.lite.Interpreter(model_path="hand_landmark.tflite") # ouput is 21*2
# path = "/home/dipesh/Hackathon-where-is-my-sanitizer/tflite/models/hand_landmark.tflite"
path = "/home/dipesh/Hackathon-where-is-my-sanitizer/tflite/models/hand_landmark_3d.tflite"
interpreter = tf.lite.Interpreter(model_path=path) # ouput is 21*2
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.

# print(input_details)
# print(output_details)




# input_shape = input_details[0]['shape']
# input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
# interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.set_tensor(input_details[0]['index'], frame)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
print(output_data.shape)

# for i in range(21):

# x = output_data[0,:21]
# y = output_data[0,21:42]
# z = output_data[0,42:]
x = [output_data[0][i*2] for i in range(21)]
y = [output_data[0][i*2 + 1] for i in range(21)]
x = np.asarray(x, dtype = np.float32)
y = np.asarray(y, dtype = np.float32)
print(x.shape)
print(y.shape)
# print(z.shape)

triang = tri.Triangulation(x, y)

# Mask off unwanted triangles.
triang.set_mask(np.hypot(x[triang.triangles].mean(axis=1), y[triang.triangles].mean(axis=1)))

plt.imshow(frame)
fig1, ax1 = plt.subplots()
ax1.set_aspect('equal')
ax1.triplot(triang, 'bo-', lw=1)
ax1.set_title('triplot of Delaunay triangulation')      
# imgplot = plt.imshow(frame)

plt.show()