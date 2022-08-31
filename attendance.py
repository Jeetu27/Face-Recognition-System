import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import json

path = 'images'
images = []
personName = []
myList = os.listdir(path)
print(myList)

# split the name and extensions
for cu_img in myList:
    current_Img = cv2.imread(f'{path}/{cu_img}')
    images.append(current_Img)
    personName.append(os.path.splitext(cu_img)[0])
print(personName)


# creating face encodings using a function
def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = faceEncodings(images)
print("All encodings created!")

# hog algorithm is used to do this encoding


# creating Function for marking attendance
def attendance(name):
    with open('attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            final_add_list = []
            final_add_list.append(name);
            final_add_list.append(tStr);
            final_add_list.append(dStr);
            json_string = json.dumps(final_add_list)
            print(json_string)
           # print(final_add_list)
            f.writelines(f'{name},{tStr},{dStr}')


# code for reading camera
cap = cv2.VideoCapture(0)  # if we are using webcam then we should put '1' in id otherwise '0'

while True:
    ret, frame = cap.read()
    # setting the format of the all cameras
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    # finding the face location through camera
    facesCurrentFrame = face_recognition.face_locations(faces)
    # finding encodings of faces
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

    # verifying the faces and finding out the faces location
    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDistance = face_recognition.face_distance(encodeListKnown, encodeFace)

        # finding minimum distance
        match_Index = np.argmin(faceDistance)

        if matches[match_Index]:
            name = personName[match_Index].upper()
            # print(name)

            # creating rectangle on the face
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

            # Calling attendance function
            attendance(name)

    cv2.imshow("Camera", frame)

    # we need to press enter key to close the camera (here '13' is ASCII value of enter)
    if cv2.waitKey(10) == 13:
        break

cap.release()
cv2.destroyAllWindows()
