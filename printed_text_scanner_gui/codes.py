import sys
import os
import cv2
import numpy as np
from PIL import Image
import pytesseract

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel,
    QTextEdit, QFileDialog, QHBoxLayout, QVBoxLayout,
    QWidget, QMessageBox, QFrame
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QTimer, QRect


# -------------------------------
#   LABEL WITH DRAWABLE ROI
# -------------------------------
class ImageLabel(QLabel):
    """QLabel that lets the user draw an ROI rectangle."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Box)
        self.pix = None
        self.start_pos = None
        self.end_pos = None
        self._drawing = False
        self.roi_rect = None  # (x,y,w,h) in image coordinates

    def setPixmap(self, pm: QPixmap):
        super().setPixmap(pm)
        self.pix = pm

    def mousePressEvent(self, event):
        if self.pix is None:
            return
        if event.button() == Qt.LeftButton:
            self._drawing = True
            self.start_pos = event.pos()
            self.end_pos = self.start_pos
            self.update()

    def mouseMoveEvent(self, event):
        if self._drawing:
            self.end_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._drawing:
            self._drawing = False
            self.end_pos = event.pos()
            self._compute_roi()
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        pen = QPen(QColor(0, 255, 0), 2)
        painter.setPen(pen)

        # live drawing rectangle
        if self._drawing and self.start_pos and self.end_pos:
            r = QRect(self.start_pos, self.end_pos).normalized()
            painter.drawRect(r)

        # saved ROI
        if self.roi_rect:
            pen2 = QPen(QColor(255, 0, 0), 2, Qt.DashLine)
            painter.setPen(pen2)
            x, y, w, h = self._image_to_display(self.roi_rect)
            painter.drawRect(QRect(int(x), int(y), int(w), int(h)))

    # -------------------------
    # Coordinate conversions
    # -------------------------

    def _compute_roi(self):
        if not self.pix:
            return
        x1, y1 = self.start_pos.x(), self.start_pos.y()
        x2, y2 = self.end_pos.x(), self.end_pos.y()

        rx, ry, rw, rh = QRect(x1, y1, x2 - x1, y2 - y1).normalized().getRect()
        self.roi_rect = self._display_to_image(rx, ry, rw, rh)

    def _display_to_image(self, x, y, w, h):
        if self.pix is None:
            return 0, 0, 0, 0

        label_w, label_h = self.width(), self.height()
        pix_w, pix_h = self.pix.width(), self.pix.height()

        scale = max(pix_w / label_w, pix_h / label_h)

        ix = int(x * scale)
        iy = int(y * scale)
        iw = int(w * scale)
        ih = int(h * scale)

        ix = max(0, ix)
        iy = max(0, iy)
        iw = max(0, min(iw, pix_w - ix))
        ih = max(0, min(ih, pix_h - iy))

        return ix, iy, iw, ih

    def _image_to_display(self, rect):
        if not rect or not self.pix:
            return 0, 0, 0, 0

        x, y, w, h = rect
        pix_w, pix_h = self.pix.width(), self.pix.height()
        label_w, label_h = self.width(), self.height()

        scale = 1.0 / max(pix_w / label_w, pix_h / label_h)
        return x * scale, y * scale, w * scale, h * scale


# -------------------------------
#      MAIN OCR APPLICATION
# -------------------------------
class OCRApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Printed Text Scanner (PyTesseract + PyQt5)")

        self.image = None
        self.display_image = None
        self.capture = None

        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)

        # UI Layout
        self.image_label = ImageLabel()
        self.image_label.setFixedSize(640, 480)

        # Buttons
        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)

        cam_start = QPushButton("Start Camera")
        cam_start.clicked.connect(self.start_camera)

        cam_stop = QPushButton("Stop Camera")
        cam_stop.clicked.connect(self.stop_camera)

        capture = QPushButton("Capture Frame")
        capture.clicked.connect(self.capture_frame)

        ocr_btn = QPushButton("Run OCR")
        ocr_btn.clicked.connect(self.run_ocr)

        clear_btn = QPushButton("Clear ROI")
        clear_btn.clicked.connect(self.clear_roi)

        save_btn = QPushButton("Save Overlay")
        save_btn.clicked.connect(self.save_overlay)

        # Text output area
        self.text_area = QTextEdit()
        self.text_area.setPlaceholderText("Extracted text will appear here...")

        # Layout arrangement
        button_row = QHBoxLayout()
        for b in [load_btn, cam_start, cam_stop, capture, ocr_btn, clear_btn, save_btn]:
            button_row.addWidget(b)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.image_label)
        left_layout.addLayout(button_row)

        main = QHBoxLayout()
        main.addLayout(left_layout)
        main.addWidget(self.text_area)

        wrapper = QWidget()
        wrapper.setLayout(main)
        self.setCentralWidget(wrapper)

    # -------------------------------
    # IMAGE LOADING
    # -------------------------------
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return

        bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            QMessageBox.warning(self, "Error", "Failed to load image.")
            return

        self.image = bgr
        self._show_image(self.image)

    # -------------------------------
    # CAMERA CONTROL
    # -------------------------------
    def start_camera(self):
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            QMessageBox.warning(self, "Camera", "Could not open camera.")
            return
        self.timer.start(30)

    def stop_camera(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.capture:
            self.capture.release()
            self.capture = None

    def _update_frame(self):
        if not self.capture:
            return
        ret, frame = self.capture.read()
        if ret:
            self.image = frame.copy()
            self._show_image(self.image)

    def capture_frame(self):
        if self.image is None:
            QMessageBox.information(self, "Info", "No frame to capture.")
            return
        QMessageBox.information(self, "Captured", "Frame captured. Now run OCR.")

    # -------------------------------
    # DISPLAY IMAGE
    # -------------------------------
    def _show_image(self, bgr, overlay_boxes=None, overlay_texts=None):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        disp = rgb.copy()

        if overlay_boxes:
            for (x, y, w, h), t in zip(overlay_boxes, overlay_texts):
                cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(disp, t, (x, max(10, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        self.display_image = disp
        h, w, ch = disp.shape
        qt_img = QImage(disp.data, w, h, ch * w, QImage.Format_RGB888)

        pix = QPixmap.fromImage(qt_img).scaled(self.image_label.width(),
                                               self.image_label.height(),
                                               Qt.KeepAspectRatio)
        self.image_label.setPixmap(pix)

    # -------------------------------
    # OCR PROCESSING
    # -------------------------------
    def run_ocr(self):
        if self.image is None:
            QMessageBox.information(self, "Info", "Load or capture an image first.")
            return

        roi = self.image_label.roi_rect
        target = self.image[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]] if roi else self.image

        if target.size == 0:
            QMessageBox.warning(self, "Error", "Invalid ROI.")
            return

        # Preprocessing
        gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        th = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 12)
        proc = th

        pil_img = Image.fromarray(proc)
        config = r"--oem 3 --psm 6"

        try:
            data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT, config=config)
            text = pytesseract.image_to_string(pil_img, config=config)
        except Exception as e:
            QMessageBox.critical(self, "Tesseract Error", str(e))
            return

        # Parse OCR boxes safely
        boxes = []
        texts = []

        for i in range(len(data["text"])):
            raw_conf = data["conf"][i]
            try:
                conf = int(raw_conf)
            except:
                conf = -1

            if conf > 40:
                txt = data["text"][i].strip()
                if txt:
                    x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                    boxes.append((x, y, w, h))
                    texts.append(txt)

        # Remap ROI boxes to full image
        if roi:
            ox, oy, _, _ = roi
            boxes = [(x + ox, y + oy, w, h) for (x, y, w, h) in boxes]

        # Show overlay
        display_copy = self.image.copy()
        self._show_image(display_copy, boxes, texts)

        # Output text to sidebar
        self.text_area.setPlainText(text)

    # -------------------------------
    # OTHER UTILITY FUNCTIONS
    # -------------------------------
    def clear_roi(self):
        self.image_label.roi_rect = None
        self.image_label.update()

    def save_overlay(self):
        if self.display_image is None:
            QMessageBox.information(self, "Info", "Nothing to save.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG (*.png)")
        if not path:
            return

        bgr = cv2.cvtColor(self.display_image, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".png", bgr)
        buf.tofile(path)

        QMessageBox.information(self, "Saved", f"Overlay saved to:\n{path}")


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OCRApp()
    window.show()
    sys.exit(app.exec_())