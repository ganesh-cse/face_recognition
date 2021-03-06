import numpy as np
import pandas
import cv2
import face_recognition
from datetime import datetime


first_frame=None
status_list=[None,None]
times=[]
df=pandas.DataFrame(columns=["Name","Start","End"])


video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Joe Biden"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
face_recog = []
name = "Unknow"

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    
    status=0
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(21,21),0)
    
    if first_frame is None:
        first_frame=gray
        continue
    
    delta_frame=cv2.absdiff(first_frame,gray)
    thresh_frame=cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame=cv2.dilate(thresh_frame, None, iterations=2)

    (cnts,_)=cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in cnts:
        if cv2.contourArea(contour) < 100000:
            continue
        status=1

        (x, y, w, h)=cv2.boundingRect(contour)
        
    status_list.append(status)
    status_list=status_list[-2:]
    
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    
    for face_encoding in face_encodings:
        name = "Unkown"
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            break;
            

    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
        face_recog.append(name)
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())
        face_recog.append(name)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_recog):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        name = "Unknown"
        
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        if status==1:
            times.append(datetime.now())
        break

for i in range(0,len(times),2):
    df=df.append({"Name":face_recog[i],"Start":times[i],"End":times[i+1]},ignore_index=True)
    
df.to_csv("Times.csv")

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
