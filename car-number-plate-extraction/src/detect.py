import cv2
import numpy as np

MIN_AREA = 600
AR_MIN, AR_MAX = 2.0, 8.0

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

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not opened")
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        vis = frame.copy()
        candidates = find_plate_candidates(frame)
        
        msg = "Searching for plate..."
        color = (0, 200, 255)
        
        if candidates:
            msg = "Plate detected"
            color = (0, 255, 0)
            for rect in candidates:
                box = cv2.boxPoints(rect).astype(int)
                cv2.polylines(vis, [box], True, (0, 255, 0), 2)
        
        # Moved these outside the if and for loops
        cv2.putText(
            vis,
            msg,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )
        cv2.putText(
            vis,
            "Press q to quit",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        cv2.imshow("Plate Detection", vis)
        
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()