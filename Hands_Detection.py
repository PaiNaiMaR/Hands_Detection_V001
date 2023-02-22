"""
 Hand Detection
 Mr.Tawatchai Poungmanee
 Class-Audio@Outlook.co.th
 Library
 : opencv2
 : mediapipe
"""
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# SET webcam input:
cap = cv2.VideoCapture(0)
if cap.isOpened():
  print("can open camera")
else:
  cap = cv2.VideoCapture(cv2.CAP_V4L2)
  print("can open camera")

with mp_hands.Hands(
  model_complexity=0,
  min_detection_confidence=0.6,  # 0 - 1 
  min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("......")
      continue
    image = cv2.flip(image, 1)
    image.flags.writeable = False
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    fingerCount = 0
    L = 0
    R = 0
    H = 1
    
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        handIndex = results.multi_hand_landmarks.index(hand_landmarks)
        handLabel = results.multi_handedness[handIndex].classification[0].label
        
        handLandmarks = []

        for landmarks in hand_landmarks.landmark:
          handLandmarks.append([landmarks.x, landmarks.y])

        if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
          fingerCount = fingerCount + 1
        elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
          fingerCount = fingerCount + 1
        
        if handLandmarks[8][1] < handLandmarks[6][1]:
          fingerCount = fingerCount + 1
        if handLandmarks[12][1] < handLandmarks[10][1]:
          fingerCount = fingerCount + 1
        if handLandmarks[16][1] < handLandmarks[14][1]:
          fingerCount = fingerCount + 1
        if handLandmarks[20][1] < handLandmarks[18][1]:
          fingerCount = fingerCount + 1

        mp_drawing.draw_landmarks(
          image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style()
        )
        if handLabel == "Left":
          L = 1
        elif handLabel == "Right":
          R = 2
          
    #Hand L - R
    H = L + R 
    if H == 1:
      cv2.putText(image, "Left", (500,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 2)
    elif H == 2:
      cv2.putText(image, "Right", (500,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 2)
    elif H == 3:
      cv2.putText(image, "Left-Right", (500,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 2)

    if results.multi_hand_landmarks == None:
        H = 0 
    
    cv2.putText(image, str(fingerCount), (20,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0,255), 3)
    cv2.imshow("FingerCounting Apps", image)
    #Exit Key ESC
    if cv2.waitKey(1) & 0xFF == 27:
      break
  cap.release()
