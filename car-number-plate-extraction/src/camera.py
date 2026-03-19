import cv2
cap = cv2.VideoCapture(0)
if not cap.isOpened():
   raise RuntimeError("Camera not opened")
while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow("Camera Test", frame)
    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()