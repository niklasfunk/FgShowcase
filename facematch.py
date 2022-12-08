import cv2
from deepface import DeepFace
import numpy as np
import time

# import pre-trained classifiers from openCV
face_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'

face_cascade = cv2.CascadeClassifier()

if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print("Error loading face cascade file")

# getting input from webcam (0 as index should open default camera of device)
video_input = cv2.VideoCapture(-1)

# counter
# img_counter = 0

# check if videofeed is processed/used
while video_input.isOpened():
    _,frame = video_input.read()

    

    # applying grayscale to video input for better results
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = face_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5)

    # create rectangle and position it around recognized face
    for x, y, w, h in face:
        image = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

    # display output
    cv2.imshow('video', frame)

    key = cv2.waitKey(1)

    # take picture and start analyzing if space key is pressed
    if(key%256 == 32):
    #    img_name = "./images/opencvframe{}.png".format(img_counter)  --> can be used to save all images with ascending filenames
        img_name = "./images/tempimg.png"
        cv2.imwrite(img_name, frame)
        img_analyze = cv2.imread(img_name)
        print("...analyzing image")
        

        try: 
            result = DeepFace.analyze(img_analyze, actions = ['emotion'])
            print("---------------------")
            print("---------------------")
            print(result['dominant_emotion']) # print dominant recognized emotion

        except: 
            print("---------------------")
            print("---------------------")
            print("no face detected") 
        
        print("---------------------")
        print("---------------------")

    #    img_counter += 1

    # specify which key breaks the loop and stops the process
    # if(cv2.waitKey(1) & 0xFF == ord('q')):
    # exit loop if escape key is pressed
    if(key%256 == 27):
        break
        
video_input.release()
