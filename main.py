import cv2
from random import randrange

traning_data = cv2.CascadeClassifier('frontal_face_default.xml')

web_cam = cv2.VideoCapture(0)

while True:
    successfull_frame_read, frame = web_cam.read()
    gray_cam = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #detect face
    face_coordinates = traning_data.detectMultiScale(frame)
    print(face_coordinates)
    
    #drow ractangle shape on pic
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame,(x, y), (x+w, y+h), (0, 255,0, 10))
    cv2.imshow('Face Detection System',frame)
    key = cv2.waitKey(1)
    if key==81 or key==113:
        break
web_cam.release()

print("Code Completed Successfully !!!")
