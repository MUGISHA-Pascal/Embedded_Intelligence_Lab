import cv2
import numpy as np
import pytesseract
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
25
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
config='--psm 8 --oem 3 -c
tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
)
return text.strip().replace(" ", ""), thresh
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
plate_img = None
thresh = None
26
if candidates:
rect = max(candidates, key=lambda r: r[1][0] * r[1][1])
box = cv2.boxPoints(rect).astype(int)
# draw bounding box
cv2.polylines(vis, [box], True, (0, 255, 0), 2)
# OCR first
plate_img = warp_plate(frame, rect)
plate_text, thresh = read_plate_text(plate_img)
msg = "OCR running"
color = (0, 255, 0)
# annotate extracted text at bottom-right of bbox
if plate_text:
x = int(np.max(box[:, 0]))
y = int(np.max(box[:, 1])) + 25
# keep text inside frame
x = min(x, vis.shape[1] - 200)
y = min(y, vis.shape[0] - 10)
cv2.putText(
vis,
plate_text,
(x, y),
cv2.FONT_HERSHEY_SIMPLEX,
0.8,
(0, 255, 0),
2
)
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
cv2.imshow("OCR Test", vis)
if plate_img is not None:
cv2.imshow("Aligned Plate", plate_img)
if thresh is not None:
cv2.imshow("Thresholded Plate", thresh)
27
if (cv2.waitKey(1) & 0xFF) == ord("q"):
break
cap.release()
cv2.destroyAllWindows()
if __name__ == "__main__":
main()