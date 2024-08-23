from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import playsound
import time
from threading import Thread
import imutils
import time
import dlib
import cv2
import numpy as np
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from lane import *
import streamlit as st
import requests
import pandas as pd
from twilio.rest import Client

API_KEY = "GOOGLE_API_KEY"
account_sid = 'ACCOUNT_SID'
auth_token = 'AUTH_TOKEN'

client = Client(account_sid, auth_token)

st.title("Intelligent Driver Assistance System")
stframe = st.empty()
col1, col2 = st.columns(2)

if 'running' not in st.session_state:
    st.session_state.running = False

def start_button():
    st.session_state.running = True

def quit_button():
    st.session_state.running = False

start= st.button("Start your Drive", on_click=start_button)
quit = st.button("Drive Over", on_click=quit_button)

image_placeholder1 = col1.empty()
image_placeholder2 = col2.empty()

def sound_alarm(path):
    playsound.playsound(path)

def get_location():
    url = f'https://www.googleapis.com/geolocation/v1/geolocate?key={API_KEY}'
    response = requests.post(url)
    if response.status_code == 200:
        location = response.json().get('location')
        lat, lng = location.get('lat'), location.get('lng')
        return lat, lng
    else:
        st.error('Error fetching location')
        return None, None

#loading facial camera detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

alarm_beep = "beep-02.wav"
alarm_yawn = "take_a_break.wav"
alarm_on = False
alarm_on_yawn = False
counter_ear = 0
counter_mar = 0
both_count = 0

#web cam dimensions
road_width = 1280
road_height = 720
road_dimensions = (road_width, road_height)

#mobile camera dimensions
frame_width = 1024
frame_height = 576

image_points = np.array([
    (359, 391),     # Nose tip 34
    (399, 561),     # Chin 9
    (337, 297),     # Left eye left corner 37
    (513, 301),     # Right eye right corne 46
    (345, 465),     # Left Mouth corner 49
    (453, 469)      # Right mouth corner 55
], dtype="double")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# grab the indexes of the facial landmarks for the mouth
(mStart, mEnd) = (49, 68)

# while True and road_cam.isOpened():
while True:
    if st.session_state.running:
        vs = VideoStream(src=1).start()
        time.sleep(2.0)
        road_cam = cv2.VideoCapture(0)
        placeholder=st.empty()
        placeholder.text("All Okay")
    
        while st.session_state.running:
            road_ret,road_frame = road_cam.read()
            resized_road_frame = cv2.resize(road_frame, road_dimensions, interpolation=cv2.INTER_AREA)
            flipped_road_frame = cv2.flip(resized_road_frame, -1)

            processed_road_frame, lane_departure, road_version_frame = process_frame(flipped_road_frame)
            #cv2.imshow('Road Frame', processed_road_frame)

            # bigger_road_frame = cv2.resize(road_version_frame, (1280, 720))
            # cv2.imshow('Road 2 Frame', bigger_road_frame)

            # grab the frame from the threaded video stream, resize it to
            # have a maximum width of 400 pixels, and convert it to
            # grayscale
            frame = vs.read()
            frame = imutils.resize(frame, width=720, height=280) #originally: 1024, 576
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            size = gray.shape

            # detect faces in the grayscale frame
            rects = detector(gray, 0) #frontal face detector 

            # check to see if a face was detected, and if so, draw the total
            # number of faces on the frame
            if len(rects) > 0:
                text = "{} face(s) found".format(len(rects))
                cv2.putText(frame, text, (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # loop over the face detections
            for rect in rects:
                # compute the bounding box of the face and draw it on the frame
                (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
                cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)

                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                #EYES
                # extract the left and right eye coordinates
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)

                # average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (250, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # compute the convex hull for the left and right eye, then visualize each of the eyes
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                #MOUTH
                mouth = shape[mStart:mEnd]
                mouthMAR = mouth_aspect_ratio(mouth)
                mar = mouthMAR

                # compute the convex hull for the mouth, then visualize the mouth
                mouthHull = cv2.convexHull(mouth)
                cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
                cv2.putText(frame, "MAR: {:.2f}".format(mar), (550, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if mar > 0.83 and mar < 0.92:
                    counter_mar += 1
                    print(counter_mar)
                    cv2.putText(frame, "YAWNING", (450,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if counter_mar > 20:
                    print("TAKE A BREAK")
                    cv2.putText(frame, "TAKE A BREAK", (450,70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    counter_mar = 0
                    if not alarm_on_yawn:
                        alarm_on_yawn = True
                        t = Thread(target = sound_alarm, args=(alarm_yawn, ))
                        t.daemon = True
                        t.start()
                    alarm_on_yawn = False
                    
                if ear < 0.25:
                    counter_ear += 1
                else:
                    counter_ear = 0
            
                if counter_ear >= 10:
                    if not alarm_on:
                        alarm_on = True
                        t = Thread(target = sound_alarm, args=(alarm_beep, ))
                        t.daemon = True
                        t.start()
                    alarm_on = False
            
                if lane_departure and counter_ear >= 10:
                    both_count += 1
                    placeholder.text("SOS!")

                if both_count > 5:
                    placeholder.text("RED ALERT")
                    lat, lng = get_location()
                    if lat and lng:
                        # st.write(f'Latitude: {lat}')
                        # st.write(f'Longitude: {lng}')
                        # st.map(pd.DataFrame({'lat': [lat], 'lon': [lng]}))
                        st.success("Location sent successfully.")
                    else:
                        st.error('Unable to fetch location')
                    both_count = 0
                    maps_link=f"https://www.google.com/maps?q={lat},{lng}"
                    message = client.messages.create(
                    body='Contact Driver Name!! They might be drowsy!\nHere is the live location of Driver Name\n'+maps_link,
                    from_='whatsapp:+14155238886',  # This is the Twilio Sandbox number for WhatsApp
                    to='whatsapp:+918850867084'       # Recipient's WhatsApp number
                    )
                    st.write(f"Message sent with SID: {message.sid}")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_road_frame_rgb = cv2.cvtColor(processed_road_frame, cv2.COLOR_BGR2RGB)

            image_placeholder1.image(frame_rgb, channels="RGB", caption="Face Camera")
            image_placeholder2.image(processed_road_frame_rgb, channels="RGB", caption="Road Camera")

        # do a bit of cleanup
        road_cam.release()
        cv2.destroyAllWindows()
        vs.stop()

    if st.session_state.running == False:
        break
    