import cv2
import numpy as np
import pytesseract
import re
import csv
import os
import time
from collections import Counter

MIN_AREA = 600
AR_MIN, AR_MAX = 2.0, 8.0
W_OUT, H_OUT = 450, 140

PLATE_RE = re.compile(r'[A-Z]{3}[0-9]{3}[A-Z]')
BUFFER_SIZE = 5
COOLDOWN = 10  # seconds

csv_file = "plates_log.csv"

# Create CSV if not exists
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Plate Number", "Timestamp"])


def find_plate_candidates(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 100, 200)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        rect = cv2.minAreaRect(cnt)
        (_, _), (w, h), _ = rect

        if w <= 0 or h <= 0:
            continue

        ar = max(w, h) / max(1.0, min(w, h))

        if AR_MIN <= ar <= AR_MAX:
            candidates.append(rect)

    return candidates


def order_points(pts):
    pts = np.array(pts, dtype=np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array(
        [top_left, top_right, bottom_right, bottom_left],
        dtype=np.float32
    )


def warp_plate(frame, rect):
    box = cv2.boxPoints(rect)
    src = order_points(box)

    dst = np.array([
        [0, 0],
        [W_OUT - 1, 0],
        [W_OUT - 1, H_OUT - 1],
        [0, H_OUT - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, M, (W_OUT, H_OUT))

    return warped


def read_plate_text(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    text = pytesseract.image_to_string(
        thresh,
        config='--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    )

    text = text.upper().replace(" ", "").replace("-", "")
    return text


def extract_valid_plate(text):
    m = PLATE_RE.search(text)
    if m:
        return m.group(0)
    return None


def majority_vote(buffer):
    if not buffer:
        return None
    return Counter(buffer).most_common(1)[0][0]


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Camera not opened")

    plate_buffer = []
    last_saved_plate = None
    last_saved_time = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        vis = frame.copy()
        candidates = find_plate_candidates(frame)

        if candidates:
            rect = max(candidates, key=lambda r: r[1][0] * r[1][1])
            box = cv2.boxPoints(rect).astype(int)

            cv2.polylines(vis, [box], True, (0, 255, 0), 2)

            plate_img = warp_plate(frame, rect)
            raw_text = read_plate_text(plate_img)
            valid_plate = extract_valid_plate(raw_text)

            if valid_plate:
                plate_buffer.append(valid_plate)

                if len(plate_buffer) > BUFFER_SIZE:
                    plate_buffer.pop(0)

                confirmed_plate = majority_vote(plate_buffer)

                x = int(np.max(box[:, 0])) - 300
                y = int(np.max(box[:, 1])) + 25

                cv2.putText(
                    vis,
                    f"CONFIRMED: {confirmed_plate}",
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                now = time.time()

                if (
                    confirmed_plate
                    and confirmed_plate != last_saved_plate
                    and (now - last_saved_time) > COOLDOWN
                ):
                    with open(csv_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            confirmed_plate,
                            time.strftime("%Y-%m-%d %H:%M:%S")
                        ])

                    print(f"[SAVED] {confirmed_plate}")

                    last_saved_plate = confirmed_plate
                    last_saved_time = now

        cv2.imshow("Temporal Validation", vis)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()