import cv2
import mediapipe as mp
import math

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            if len(lmList) != 0:
                # Thumb tip (4) and Index tip (8)
                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1], lmList[8][2]

                # Center point to place the text
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Calculate distance
                length = math.hypot(x2 - x1, y2 - y1)

                # The "6-7" Logic: If the gap is small but not closed
                if 20 < length < 90:
                    # Draw a line between the fingers
                    cv2.line(img, (x1, y1), (x2, y2), (255, 180, 203), 3)
                    # Add the meme text
                    cv2.putText(img, "6-7 alli-chuzz", (cx + 20, cy),
                                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 180, 203), 2)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Meme Detector", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
