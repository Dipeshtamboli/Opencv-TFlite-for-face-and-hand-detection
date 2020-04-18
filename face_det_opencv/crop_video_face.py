import cv2
import sys
import os
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')

video_capture = cv2.VideoCapture(1)

img_counter = 0

# img_counter =0
print("Type name of the person")
name= input()
print("Hi {}".format(name))
folder_name = "../data/{}/".format(name)
print(folder_name)
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
# exit()
def crop_resize_save(image, x,y,w,h):
    global img_counter    
    m = 50
    # print(y-m,y+h+m, x-m,x+w+m)
    y_top = y-m
    y_bottom=y+h+m
    x_left=x-m
    x_right=x+w+m
    if(y_top<0):
        y_top=1
    if(x_left<0):
        x_left=1


    cropped = image[y_top:y_bottom, x_left:x_right]
    # cropped = image[y-m:y+h+m, x-m:x+w+m]
    resized = cv2.resize(cropped,(500, 500), interpolation = cv2.INTER_AREA)
    # img_name = "IMGS/facedetect_webcam_{}.png".format(img_counter)
    img_name = folder_name + "/facedetect_webcam_{}.png".format(img_counter)
    cv2.imwrite(img_name, resized)
    # print("{} written!".format(img_name))
    img_counter = img_counter + 1

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # cv2.imshow('raw',frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k = cv2.waitKey(1)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        # print(x,y,w,h)
        crop_resize_save(frame,x,y,w,h)
        # img_counter =img_counter+1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    # Display the resulting frame
    cv2.imshow('FaceDetection', frame)

    if k%256 == 27: #ESC Pressed
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "../data/img/facedetect_webcam_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()