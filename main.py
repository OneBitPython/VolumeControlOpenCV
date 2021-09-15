import subprocess
import cv2
import mediapipe as mp
from scipy.spatial import distance as dist
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.7, max_num_hands=2)

mpDraw = mp.solutions.drawing_utils

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()


minVol = volRange[0]
maxVol = volRange[1]
while True:
    ret, frame = cap.read()

    RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(RGBframe)

    my_list = []
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            for id, lm in enumerate(hand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                my_list.append([id, cx, cy])

            mpDraw.draw_landmarks(frame, hand, mpHands.HAND_CONNECTIONS)

    if len(my_list) > 0:
        v1, v2 = my_list[4][1], my_list[4][2]
        y1, y2 = my_list[8][1], my_list[8][2]

        cx, cy = (v1 + y1) // 2, (v2 + y2) // 2

        cv2.circle(frame, (v1, v2), 8, (255, 0, 0), -1)
        cv2.circle(frame, (y1, y2), 8, (255, 0, 0), -1)
        cv2.circle(frame, (cx, cy), 8, (255, 0, 0), -1)

        cv2.line(frame, (v1, v2), (y1, y2), (0, 0, 255), 1)

        length = dist.euclidean((v1, v2), (y1, y2))
        vol = np.interp(length, [3, 85], [minVol, maxVol])
        volume.SetMasterVolumeLevel(vol, None)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()


"""import subprocess
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()


mpDraw = mp.solutions.drawing_utils
while True:
    ret, frame = cap.read()

    RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(RGBframe)

    my_list = []
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            for id, lm in enumerate(hand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)

                my_list.append([id, cx, cy])
                if id == 0:
                    cv2.circle(frame, (cx, cy), 8, (255, 0, 0), -1)

            mpDraw.draw_landmarks(frame, hand, mpHands.HAND_CONNECTIONS)



    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()


# subprocess.call(['powershell', '(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,100)'])
"""
