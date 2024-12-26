import cv2
from cv2.typing import MatLike
import mediapipe as mp

mp_hands = mp.solutions.hands # type: ignore
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils # type: ignore
finger_coordinates = [(8, 6), (12, 10), (16, 14), (20, 18)]
thumb_coordinates = (4, 2)

def count_fingers(hand_points: list) -> int:
    up_count = 0
    for coordinate in finger_coordinates:
        if hand_points[coordinate[0]][1] < hand_points[coordinate[1]][1]:
            up_count += 1
    if hand_points[thumb_coordinates[0]][0] > hand_points[thumb_coordinates[1]][0]:
        up_count += 1
    return up_count

def process_frame(frame: MatLike) -> MatLike:
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    multi_hand_landmarks = results.multi_hand_landmarks
    if multi_hand_landmarks:
        hand_points = []
        for hand_landmarks in multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for idx, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                hand_points.append((cx, cy))
        for point in hand_points:
            cv2.circle(frame, point, 10, (0, 0, 255), cv2.FILLED)
        up_count = count_fingers(hand_points)

        cv2.putText(frame, str(up_count), (150, 150), cv2.FONT_HERSHEY_PLAIN, 12, (255, 0, 0), 12)
    return frame


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        frame = process_frame(img)
        cv2.imshow("frame", frame)
        cv2.waitKey(1)