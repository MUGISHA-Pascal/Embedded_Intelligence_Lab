# src/landmarks.py
"""
Minimal pipeline:
camera -> Haar face box -> MediaPipe FaceMesh (or bbox 5pt fallback) -> extract 5 keypoints -> draw

Run:
python -m src.landmarks

Keys:
q : quit
"""

import cv2
import numpy as np

try:
    import mediapipe as mp
    _mp_ok = getattr(mp, "solutions", None) is not None
except Exception:
    _mp_ok = False
    mp = None

# 5-point indices (FaceMesh)
IDX_LEFT_EYE = 33
IDX_RIGHT_EYE = 263
IDX_NOSE_TIP = 1
IDX_MOUTH_LEFT = 61
IDX_MOUTH_RIGHT = 291


def _bbox_5pt(x, y, w, h):
    """Fallback 5pt from bbox when MediaPipe is unavailable."""
    return np.array([
        [x + 0.30 * w, y + 0.35 * h],
        [x + 0.70 * w, y + 0.35 * h],
        [x + 0.50 * w, y + 0.55 * h],
        [x + 0.35 * w, y + 0.78 * h],
        [x + 0.65 * w, y + 0.78 * h],
    ], dtype=np.float32)


def main():
    # Haar
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face = cv2.CascadeClassifier(cascade_path)
    if face.empty():
        raise RuntimeError(f"Failed to load cascade: {cascade_path}")

    # FaceMesh (optional)
    fm = None
    if _mp_ok:
        try:
            fm = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception:
            fm = None
    if fm is None:
        print("MediaPipe unavailable; using bbox-based 5pt (fallback).")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not opened. Try camera index 0/1/2.")

    print("Haar + FaceMesh 5pt (minimal). Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        # draw All haar faces (no ranking)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        kps = None
        if fm is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = fm.process(rgb)
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                idxs = [IDX_LEFT_EYE, IDX_RIGHT_EYE, IDX_NOSE_TIP, IDX_MOUTH_LEFT, IDX_MOUTH_RIGHT]
                pts = []
                for i in idxs:
                    p = lm[i]
                    pts.append([p.x * W, p.y * H])
                kps = np.array(pts, dtype=np.float32)
                if kps[0, 0] > kps[1, 0]:
                    kps[0, 0], kps[1, 0] = kps[1, 0], kps[0, 0]
                if kps[3, 0] > kps[4, 0]:
                    kps[3, 0], kps[4, 0] = kps[4, 0], kps[3, 0]
        if kps is None and len(faces) > 0:
            # Fallback: largest face bbox 5pt
            areas = [f[2] * f[3] for f in faces]
            i = int(np.argmax(areas))
            x, y, w, h = faces[i]
            kps = _bbox_5pt(x, y, w, h)

        if kps is not None:
            for (px, py) in kps.astype(int):
                cv2.circle(frame, (int(px), int(py)), 4, (0, 255, 0), -1)
            cv2.putText(frame, "5pt", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("5pt Landmarks", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()