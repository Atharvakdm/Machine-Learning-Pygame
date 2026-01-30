import cv2  # importing opencv lib
import mediapipe as mp  # importing the library with specialized trained data: mediapipe
import math  # importing math to find relative distances between fingers

cap = cv2.VideoCapture(0)  # sets up the camera
mpHands = mp.solutions.hands  # extracting the mp.Hands file via Mediapipe Library
# min_detection_confidence lets you help detect the hand (palm detection).
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils  # drawing the lines between the fingers

while True:
    success, img = cap.read()  # img is the pop-up window of the camera when you run the program.
    if not success:
        break
    img = cv2.flip(img, 1)  # flipping the camera because it is inverted (default).
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converting the contents in real-time camera to RGB format.
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        # This loop pinpoints every landmark and coordinate of your finger with a number
        for hand_idx, hand_lms in enumerate(results.multi_hand_landmarks): # gives an id for each of the hand landmarks
            lmList = []
            for id, lm in enumerate(hand_lms.landmark):
                h, w, c = img.shape
                # converting into pixel size based on image dimensions.
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([cx, cy])  # appending back into the list.

            # UNIVERSAL MAPPING (Works for both Left and Right)
            # We access the landmarks by their standard Mediapipe index
            thumb = lmList[4]
            index = lmList[8]
            middle = lmList[12]
            ring = lmList[16]
            pinky = lmList[20]

            # finding the particular distances between fingers.
            # math.hypot does sqrt(x^2+y^2)
            dist_thumb_index = math.hypot(thumb[0] - index[0], thumb[1] - index[1])  # distance should be high
            dist_thumb_middle = math.hypot(thumb[0] - middle[0], thumb[1] - middle[1])  # distance should be high
            dist_thumb_ring = math.hypot(thumb[0] - ring[0], thumb[1] - ring[1])  # distance should be high
            dist_thumb_pinky = math.hypot(thumb[0] - pinky[0], thumb[1] - pinky[1])  # distance should be high
            dist_mid_index = math.hypot(middle[0] - index[0], middle[1] - index[1])
            dist_mid_ring = math.hypot(middle[0] - ring[0], middle[1] - ring[1])  # distance should be less
            dist_mid_pinky = math.hypot(middle[0] - pinky[0], middle[1] - pinky[1])  # distance should be less
            dist_ring_pinky = math.hypot(ring[0] - pinky[0], ring[1] - pinky[1])  # distance should be less

            # sign-lang for number 1 logic:
            # Number 1: Only Index up. Thumb, Mid, Ring, Pinky are curled/touching.
            if (dist_thumb_index > 50 and dist_thumb_middle <= 30 and
                    dist_mid_ring <= 30 and dist_ring_pinky <= 30):
                cv2.putText(img, "Number: 1", (150, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 120, 255), 3)

            # Number 2: Index and Middle up (V shape). Thumb, Ring, Pinky curled.
            elif (dist_thumb_index > 50 and dist_mid_index > 30 and
                  dist_thumb_ring <= 30 and dist_ring_pinky <= 30):
                cv2.putText(img, "Number: 2", (150, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 20, 255), 3)

            # Number 3: Thumb, Index, and Middle up. Ring and Pinky curled.
            # Note: ASL '3' is unique because the thumb is extended.
            elif (dist_thumb_index > 40 and dist_thumb_middle > 40 and
                  dist_ring_pinky <= 30 and dist_thumb_ring > 40):
                cv2.putText(img, "Number: 3", (150, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (230, 20, 210), 3)

            # Number 4: All four fingers up, Thumb tucked into palm.
            elif (dist_thumb_index <= 30 and dist_mid_index > 30 and
                  dist_mid_ring > 30 and dist_ring_pinky > 30):
                cv2.putText(img, "Number: 4", (150, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (240, 0, 250), 3)

            # Number 5: All fingers and Thumb spread wide.
            elif (dist_thumb_index > 50 and dist_mid_index > 40 and
                  dist_mid_ring > 40 and dist_ring_pinky > 40):
                cv2.putText(img, "Number: 5", (150, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (240, 0, 250), 3)

            # Number 6: Thumb and Pinky touching. Index, Middle, Ring are UP.
            if (dist_thumb_pinky <= 30 and dist_thumb_index > 50 and dist_thumb_middle > 50):
                cv2.putText(img, "Number: 6", (150, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (240, 100, 220), 3)

            # Number 7: Thumb and Ring finger touching. Index, Middle, Pinky are UP.
            elif (dist_thumb_ring <= 30 and dist_thumb_index > 50 and dist_thumb_pinky > 50):
                cv2.putText(img, "Number: 7", (150, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 100, 220), 3)

            # Number 8: Thumb and Middle finger touching. Index, Ring, Pinky are UP.
            elif (dist_thumb_middle <= 30 and dist_thumb_index > 50 and dist_thumb_pinky > 50):
                cv2.putText(img, "Number: 8", (150, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 3)

            # Number 9: Thumb and Index finger touching (the "OK" sign). Middle, Ring, Pinky are UP.
            elif (dist_thumb_index <= 30 and dist_thumb_middle > 50 and dist_thumb_pinky > 50):
                cv2.putText(img, "Number: 9", (150, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0), 3)

            # Number 10: Thumbs up. All other fingers curled into the palm.
            # Note: In ASL, 10 usually involves a "shake" of the thumb,
            # but for a static image, check if thumb is UP and fingers are CLOSED.
            elif (dist_thumb_index > 50 and dist_mid_index <= 30 and
                  dist_mid_ring <= 30 and dist_ring_pinky <= 30):
                cv2.putText(img, "Number: 10", (150, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 3)


            # Specifying the color of the dots representing the fingers.
            dot_style = mpDraw.DrawingSpec(color=(80, 125, 245), thickness=-1, circle_radius=4)
            mpDraw.draw_landmarks(img, hand_lms, mpHands.HAND_CONNECTIONS, dot_style)

    cv2.imshow("Hand Tracker", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
