import numpy as np
MIN_AREA = 600
AR_MIN, AR_MAX = 2.0, 8.0
W_OUT, H_OUT = 450, 140
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
        23
[W_OUT - 1, H_OUT - 1],
[0, H_OUT - 1]
], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, M, (W_OUT, H_OUT))
    return warped
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
        best_plate = None
        msg = "Detecting plate..."
        color = (0, 200, 255)
        if candidates:
            # choose largest candidate
            rect = max(candidates, key=lambda r: r[1][0] * r[1][1])
            box = cv2.boxPoints(rect).astype(int)
            cv2.polylines(vis, [box], True, (255, 0, 0), 2)
            for (x, y) in box:
                cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)
            best_plate = warp_plate(frame, rect)
msg = "Plate aligned"
color = (0, 255, 0)
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
24
cv2.imshow("Alignment Test", vis)
if best_plate is not None:
cv2.imshow("Aligned Plate", best_plate)
if (cv2.waitKey(1) & 0xFF) == ord("q"):
break
cap.release()
cv2.destroyAllWindows()
if __name__ == "__main__":
main()