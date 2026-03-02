# main.py

import cv2
import time
import pyttsx3
from collections import deque, defaultdict
from hand_detector import HandDetector
from gesture_classifier import GestureClassifier


def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    classifier = GestureClassifier()

    engine = pyttsx3.init()
    prev_time = time.time()
    last_spoken = ""

    # smoothing state for digit outputs
    SMOOTH_WINDOW = 6
    CONF_THRESHOLD = 0.5
    window = deque(maxlen=SMOOTH_WINDOW)
    left_buf = deque(maxlen=SMOOTH_WINDOW)
    right_buf = deque(maxlen=SMOOTH_WINDOW)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        results = detector.detect_hands(frame)

        # collect detected hands' finger states and wrist x positions
        detected_hands = []
        wrist_xs = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                detector.draw_landmarks(frame, hand_landmarks)
                try:
                    wrist_xs.append(hand_landmarks.landmark[0].x)
                except Exception:
                    wrist_xs.append(None)
                fingers = classifier.get_finger_status(hand_landmarks)
                detected_hands.append(fingers)

        predicted = "UNKNOWN"
        conf = 0.0

        if detected_hands:
            if len(detected_hands) == 1:
                # single hand
                val = classifier.classify(detected_hands[0])
                if val != "UNKNOWN":
                    predicted = val
                    conf = 1.0
            else:
                # two hands: order by wrist x when available
                try:
                    if len(wrist_xs) >= 2 and wrist_xs[0] is not None and wrist_xs[1] is not None:
                        pairs = list(zip(wrist_xs, detected_hands))
                        pairs.sort(key=lambda x: x[0])
                        left = pairs[0][1]
                        right = pairs[1][1]
                    else:
                        left, right = detected_hands[0], detected_hands[1]

                    left_cnt = left.count(1)
                    right_cnt = right.count(1)
                    left_buf.append(left_cnt)
                    right_buf.append(right_cnt)
                    from collections import Counter
                    left_mode = Counter(left_buf).most_common(1)[0][0] if left_buf else left_cnt
                    right_mode = Counter(right_buf).most_common(1)[0][0] if right_buf else right_cnt
                    value = left_mode + right_mode
                    value = max(0, min(10, value))
                    predicted = str(value)
                    conf = 1.0
                    cv2.putText(frame, f"Left count: {left_mode}", (300, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv2.putText(frame, f"Right count: {right_mode}", (300, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                except Exception:
                    predicted = "UNKNOWN"

        # smoothing: push (predicted,conf) dict into window
        window.append({predicted: conf})

        # aggregate mean confidences
        agg = defaultdict(float)
        for d in window:
            for k, v in d.items():
                agg[k] += v
        if window:
            for k in list(agg.keys()):
                agg[k] /= len(window)

        if agg:
            best_label, best_conf = max(agg.items(), key=lambda x: x[1])
        else:
            best_label, best_conf = "UNKNOWN", 0.0

        display_label = best_label if best_conf >= CONF_THRESHOLD else "UNKNOWN"

        cv2.putText(frame, f"Digit: {display_label}", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,0,255), 3)

        if display_label != "UNKNOWN" and display_label != last_spoken:
            engine.say(display_label)
            engine.runAndWait()
            last_spoken = display_label

        # FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if current_time != prev_time else 0
        prev_time = current_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

        cv2.imshow("ISL Word Interpreter", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()