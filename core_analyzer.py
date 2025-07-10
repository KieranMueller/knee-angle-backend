import os

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def process_video(video_path: str, leg: str = "right", draw_debug: bool = False, save_debug_path: str = None,
                  output_image_path=None):
    cap = cv2.VideoCapture(video_path)
    max_angle = -1
    max_frame = None
    frames = 0

    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        return {
            "max_knee_angle": None,
            "frames_analyzed": 0,
            "output_image_path": None,
            "debug_video_path": None
        }

    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    height, width = frame.shape[:2]

    if save_debug_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_debug_path, fourcc, 30.0, (width, height))
    else:
        out = None

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    with mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Could cause issues with landscape vids, test
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                side = mp_pose.PoseLandmark
                if leg == "left":
                    hip = [lm[side.LEFT_HIP.value].x, lm[side.LEFT_HIP.value].y]
                    knee = [lm[side.LEFT_KNEE.value].x, lm[side.LEFT_KNEE.value].y]
                    ankle = [lm[side.LEFT_ANKLE.value].x, lm[side.LEFT_ANKLE.value].y]
                    if (
                            lm[side.LEFT_HIP.value].visibility < 0.5 or
                            lm[side.LEFT_KNEE.value].visibility < 0.5 or
                            lm[side.LEFT_ANKLE.value].visibility < 0.5
                    ):
                        continue
                else:
                    hip = [lm[side.RIGHT_HIP.value].x, lm[side.RIGHT_HIP.value].y]
                    knee = [lm[side.RIGHT_KNEE.value].x, lm[side.RIGHT_KNEE.value].y]
                    ankle = [lm[side.RIGHT_ANKLE.value].x, lm[side.RIGHT_ANKLE.value].y]
                    if (
                            lm[side.RIGHT_HIP.value].visibility < 0.5 or
                            lm[side.RIGHT_KNEE.value].visibility < 0.5 or
                            lm[side.RIGHT_ANKLE.value].visibility < 0.5
                    ):
                        continue

                angle = calculate_angle(hip, knee, ankle)

                if angle > max_angle:
                    max_angle = angle
                    max_frame = frame.copy()
                    max_landmarks = results.pose_landmarks
                frames += 1

                if draw_debug:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    h, w = frame.shape[:2]
                    cv2.putText(frame, f"{int(angle)}°", (int(knee[0] * w), int(knee[1] * h)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if draw_debug:
                if out:
                    out.write(frame)
                else:
                    cv2.imshow("Pose Debug", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

    if max_frame is not None and output_image_path:
        mp_drawing.draw_landmarks(
            max_frame, max_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

        # Draw the max angle text near the knee
        h, w = max_frame.shape[:2]
        knee_coords = lm[side.RIGHT_KNEE.value if leg == "right" else side.LEFT_KNEE.value]
        x, y = int(knee_coords.x * w), int(knee_coords.y * h)

        cv2.putText(
            max_frame,
            f"Max Angle: {int(max_angle)}°",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imwrite(output_image_path, max_frame)

    cap.release()
    if out:
        out.release()
        print(f"Debug video saved to {save_debug_path}, size: {os.path.getsize(save_debug_path)} bytes")
    if draw_debug and not out:
        cv2.destroyAllWindows()

    return {
        "max_knee_angle": max_angle,
        "frames_analyzed": frames,
        "output_image_path": output_image_path,
        "debug_video_path": save_debug_path
    }
