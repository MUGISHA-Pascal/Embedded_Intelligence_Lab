import sys
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageQt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QLabel, QTextEdit, QFileDialog, 
                             QSplitter, QGroupBox, QCheckBox, QSlider, QSpinBox,
                             QComboBox, QProgressBar, QMessageBox, QTabWidget)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QRect
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QFont
import os

class ImageLabel(QLabel):
    """Custom QLabel for image display with ROI selection capability"""
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 300)
        self.setStyleSheet("border: 2px solid gray;")
        self.setAlignment(Qt.AlignCenter)
        self.setText("Load an image or start camera")
        
        # ROI selection variables
        self.roi_start = None
        self.roi_end = None
        self.roi_rect = None
        self.drawing = False
        self.original_pixmap = None
        
    def set_image(self, pixmap):
        """Set image and store original for ROI operations"""
        self.original_pixmap = pixmap
        self.setPixmap(pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
    def mousePressEvent(self, event):
        """Start ROI selection"""
        if event.button() == Qt.LeftButton and self.original_pixmap:
            self.roi_start = event.pos()
            self.drawing = True
            
    def mouseMoveEvent(self, event):
        """Update ROI selection"""
        if self.drawing and self.roi_start:
            self.roi_end = event.pos()
            self.update_roi_display()
            
    def mouseReleaseEvent(self, event):
        """Finish ROI selection"""
        if event.button() == Qt.LeftButton and self.drawing:
            self.roi_end = event.pos()
            self.drawing = False
            self.update_roi_display()
            
    def update_roi_display(self):
        """Update display with ROI rectangle"""
        if not self.original_pixmap or not self.roi_start or not self.roi_end:
            return
            
        # Create a copy of the original pixmap
        pixmap_copy = self.original_pixmap.copy()
        painter = QPainter(pixmap_copy)
        
        # Draw ROI rectangle
        pen = QPen(Qt.red, 3)
        painter.setPen(pen)
        
        # Calculate rectangle coordinates
        x1, y1 = self.roi_start.x(), self.roi_start.y()
        x2, y2 = self.roi_end.x(), self.roi_end.y()
        
        # Ensure proper rectangle coordinates
        left = min(x1, x2)
        top = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        self.roi_rect = QRect(left, top, width, height)
        painter.drawRect(self.roi_rect)
        painter.end()
        
        # Display the updated pixmap
        self.setPixmap(pixmap_copy.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
    def get_roi_coordinates(self):
        """Get ROI coordinates relative to original image"""
        if not self.roi_rect or not self.original_pixmap:
            return None
            
        # Calculate scaling factors
        label_size = self.size()
        pixmap_size = self.original_pixmap.size()
        
        scale_x = pixmap_size.width() / label_size.width()
        scale_y = pixmap_size.height() / label_size.height()
        
        # Convert ROI coordinates to original image coordinates
        x = int(self.roi_rect.x() * scale_x)
        y = int(self.roi_rect.y() * scale_y)
        w = int(self.roi_rect.width() * scale_x)
        h = int(self.roi_rect.height() * scale_y)
        
        return (x, y, w, h)
        
    def clear_roi(self):
        """Clear ROI selection"""
        self.roi_start = None
        self.roi_end = None
        self.roi_rect = None
        if self.original_pixmap:
            self.setPixmap(self.original_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

class CameraThread(QThread):
    """Thread for handling camera input"""
    
    frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.camera = None
        self.running = False
        
    def start_camera(self):
        """Start camera capture"""
        self.camera = cv2.VideoCapture(0)
        if self.camera.isOpened():
            self.running = True
            self.start()
            return True
        return False
        
    def stop_camera(self):
        """Stop camera capture"""
        self.running = False
        if self.camera:
            self.camera.release()
            
    def run(self):
        """Camera capture loop"""
        while self.running and self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                self.frame_ready.emit(frame)
            self.msleep(30)  # ~30 FPS

class OCRProcessor:
    """Class for handling OCR operations"""
    
    @staticmethod
    def preprocess_image(image, enhance_contrast=True, denoise=True, threshold=True):
        """Preprocess image for better OCR results"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Enhance contrast
        if enhance_contrast:
            gray = cv2.equalizeHist(gray)
            
        # Denoise
        if denoise:
            gray = cv2.medianBlur(gray, 3)
            
        # Threshold
        if threshold:
            _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        return gray
        
    @staticmethod
    def extract_text(image, roi=None, preprocess=True):
        """Extract text from image using PyTesseract"""
        try:
            # Apply ROI if specified
            if roi:
                x, y, w, h = roi
                image = image[y:y+h, x:x+w]
                
            # Preprocess image
            if preprocess:
                processed_image = OCRProcessor.preprocess_image(image)
            else:
                processed_image = image
                
            # Extract text
            text = pytesseract.image_to_string(processed_image, config='--psm 6')
            
            # Get bounding boxes for overlay
            data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT)
            
            return text.strip(), data
            
        except Exception as e:
            return f"OCR Error: {str(e)}", None

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCR Text Scanner")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize variables
        self.current_image = None
        self.camera_thread = CameraThread()
        self.camera_active = False
        
        # Setup UI
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Image display and controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - OCR results and settings
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([700, 500])
        
    def create_left_panel(self):
        """Create left panel with image display and controls"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Image display
        self.image_label = ImageLabel()
        layout.addWidget(self.image_label)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.load_button = QPushButton("Load Image")
        self.camera_button = QPushButton("Start Camera")
        self.clear_roi_button = QPushButton("Clear ROI")
        self.ocr_button = QPushButton("Run OCR")
        
        # Style buttons
        button_style = """
            QPushButton {
                padding: 8px 16px;
                font-size: 12px;
                font-weight: bold;
                border-radius: 4px;
                border: 2px solid #3498db;
                background-color: #3498db;
                color: white;
            }
            QPushButton:hover {
                background-color: #2980b9;
                border-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """
        
        for button in [self.load_button, self.camera_button, self.clear_roi_button, self.ocr_button]:
            button.setStyleSheet(button_style)
            button_layout.addWidget(button)
            
        layout.addLayout(button_layout)
        
        return panel
        
    def create_right_panel(self):
        """Create right panel with OCR results and settings"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create tab widget
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)
        
        # OCR Results tab
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        
        results_layout.addWidget(QLabel("Extracted Text:"))
        self.text_output = QTextEdit()
        self.text_output.setFont(QFont("Courier", 10))
        results_layout.addWidget(self.text_output)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        results_layout.addWidget(self.progress_bar)
        
        tab_widget.addTab(results_tab, "OCR Results")
        
        # Settings tab
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        
        # Preprocessing options
        preprocess_group = QGroupBox("Preprocessing Options")
        preprocess_layout = QVBoxLayout(preprocess_group)
        
        self.enhance_contrast_cb = QCheckBox("Enhance Contrast")
        self.enhance_contrast_cb.setChecked(True)
        preprocess_layout.addWidget(self.enhance_contrast_cb)
        
        self.denoise_cb = QCheckBox("Denoise")
        self.denoise_cb.setChecked(True)
        preprocess_layout.addWidget(self.denoise_cb)
        
        self.threshold_cb = QCheckBox("Apply Threshold")
        self.threshold_cb.setChecked(True)
        preprocess_layout.addWidget(self.threshold_cb)
        
        settings_layout.addWidget(preprocess_group)
        
        # OCR options
        ocr_group = QGroupBox("OCR Options")
        ocr_layout = QVBoxLayout(ocr_group)
        
        self.roi_only_cb = QCheckBox("Process ROI Only")
        ocr_layout.addWidget(self.roi_only_cb)
        
        self.show_overlay_cb = QCheckBox("Show Text Overlay")
        self.show_overlay_cb.setChecked(True)
        ocr_layout.addWidget(self.show_overlay_cb)
        
        settings_layout.addWidget(ocr_group)
        
        settings_layout.addStretch()
        
        tab_widget.addTab(settings_tab, "Settings")
        
        return panel
        
    def setup_connections(self):
        """Setup signal-slot connections"""
        self.load_button.clicked.connect(self.load_image)
        self.camera_button.clicked.connect(self.toggle_camera)
        self.clear_roi_button.clicked.connect(self.clear_roi)
        self.ocr_button.clicked.connect(self.run_ocr)
        
        # Camera thread connection
        self.camera_thread.frame_ready.connect(self.update_camera_frame)
        
    def load_image(self):
        """Load image from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.gif)"
        )
        
        if file_path:
            # Load image with OpenCV
            self.current_image = cv2.imread(file_path)
            if self.current_image is not None:
                self.display_image(self.current_image)
            else:
                QMessageBox.warning(self, "Error", "Could not load image file.")
                
    def display_image(self, cv_image):
        """Display OpenCV image in QLabel"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Convert to QImage
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Convert to QPixmap and display
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.set_image(pixmap)
        
    def toggle_camera(self):
        """Toggle camera on/off"""
        if not self.camera_active:
            if self.camera_thread.start_camera():
                self.camera_active = True
                self.camera_button.setText("Stop Camera")
            else:
                QMessageBox.warning(self, "Error", "Could not start camera.")
        else:
            self.camera_thread.stop_camera()
            self.camera_active = False
            self.camera_button.setText("Start Camera")
            
    def update_camera_frame(self, frame):
        """Update display with new camera frame"""
        self.current_image = frame
        self.display_image(frame)
        
    def clear_roi(self):
        """Clear ROI selection"""
        self.image_label.clear_roi()
        
    def run_ocr(self):
        """Run OCR on current image"""
        if self.current_image is None:
            QMessageBox.warning(self, "Error", "No image loaded.")
            return
            
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Get ROI if selected
        roi = None
        if self.roi_only_cb.isChecked():
            roi = self.image_label.get_roi_coordinates()
            if roi is None:
                QMessageBox.warning(self, "Error", "No ROI selected. Please select a region or uncheck 'Process ROI Only'.")
                self.progress_bar.setVisible(False)
                return
                
        # Get preprocessing options
        enhance_contrast = self.enhance_contrast_cb.isChecked()
        denoise = self.denoise_cb.isChecked()
        threshold = self.threshold_cb.isChecked()
        
        # Process image
        image_copy = self.current_image.copy()
        
        # Apply preprocessing
        if enhance_contrast or denoise or threshold:
            processed_image = OCRProcessor.preprocess_image(
                image_copy, enhance_contrast, denoise, threshold
            )
        else:
            processed_image = image_copy
            
        # Extract text
        text, ocr_data = OCRProcessor.extract_text(processed_image, roi, preprocess=False)
        
        # Display results
        self.text_output.setPlainText(text)
        
        # Show overlay if enabled
        if self.show_overlay_cb.isChecked() and ocr_data:
            self.show_text_overlay(ocr_data, roi)
            
        # Hide progress
        self.progress_bar.setVisible(False)
        
    def show_text_overlay(self, ocr_data, roi=None):
        """Show text overlay on image"""
        if not self.current_image is not None:
            return
            
        # Create image copy for overlay
        overlay_image = self.current_image.copy()
        
        # Adjust coordinates if ROI was used
        offset_x, offset_y = 0, 0
        if roi:
            offset_x, offset_y = roi[0], roi[1]
            
        # Draw bounding boxes around detected text
        for i in range(len(ocr_data['text'])):
            if int(ocr_data['conf'][i]) > 30:  # Confidence threshold
                x = ocr_data['left'][i] + offset_x
                y = ocr_data['top'][i] + offset_y
                w = ocr_data['width'][i]
                h = ocr_data['height'][i]
                
                # Draw rectangle
                cv2.rectangle(overlay_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw text
                text = ocr_data['text'][i].strip()
                if text:
                    cv2.putText(overlay_image, text, (x, y - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                              
        # Display overlay
        self.display_image(overlay_image)

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Check if Tesseract is available
    try:
        pytesseract.get_tesseract_version()
    except Exception as e:
        QMessageBox.critical(None, "Error", 
                           f"Tesseract not found. Please install Tesseract OCR.\nError: {str(e)}")
        sys.exit(1)
        
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
