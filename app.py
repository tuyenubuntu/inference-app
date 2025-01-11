import sys
import numpy as np
import onnxruntime as ort
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QFileDialog, QWidget, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ONNX Model Inference with BBX")

        # Set fixed initial window size with 16:9 ratio, 14 inch diagonal
        self.setFixedSize(1171, 657)  # 16:9 ratio with 14 inch diagonal (96 DPI)

        # Widgets
        self.label_image = QLabel("Image will be displayed here")
        self.label_image.setAlignment(Qt.AlignCenter)
        self.label_image.setScaledContents(True)  # Allow image scaling without distortion
        self.label_image.setSizePolicy(1, 1)  # Ensure the image will not resize beyond the widget's size

        self.label_result = QLabel("Result will be displayed here")
        self.label_result.setAlignment(Qt.AlignCenter)
        self.label_result.setScaledContents(True)  # Allow image scaling without distortion
        self.label_result.setSizePolicy(1, 1)  # Ensure the image will not resize beyond the widget's size

        self.button_load = QPushButton("Load Image")
        self.button_infer = QPushButton("Run Inference")
        self.button_infer.setEnabled(False)

        # Layouts
        main_layout = QVBoxLayout()

        # Create layout for images and results
        image_layout = QHBoxLayout()

        # Layout for original image
        layout_image = QVBoxLayout()
        layout_image.addWidget(self.label_image)
        layout_image.addWidget(self.button_load)

        # Layout for result image
        layout_result = QVBoxLayout()
        layout_result.addWidget(self.label_result)
        layout_result.addWidget(self.button_infer)

        # Add image and result layouts to main layout
        image_layout.addLayout(layout_image)
        image_layout.addLayout(layout_result)

        # Add the image_layout to the main layout
        main_layout.addLayout(image_layout)

        # Set the layout for the central widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Signals
        self.button_load.clicked.connect(self.load_image)
        self.button_infer.clicked.connect(self.run_inference)

        # Model
        self.model_path = "model/aoiai_detectnet_v1.onnx"
        self.session = ort.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.input_shape = self.session.get_inputs()[0].shape
        self.image = None
        self.processed_image = None

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.bmp)")
        if file_path:
            # Load image using OpenCV
            self.image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            self.show_image(self.image)
            self.button_infer.setEnabled(True)

    def preprocess_image(self, image):
        """Preprocess the image to match model input, including cropping if necessary."""
        h, w = image.shape[:2]
        target_height = self.input_shape[2]
        target_width = self.input_shape[3]

        # Calculate scaling factor to maintain aspect ratio
        scale_w = target_width / float(w)
        scale_h = target_height / float(h)
        scale = min(scale_w, scale_h)

        # Compute the new width and height, while maintaining aspect ratio
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize the image to the new size
        resized_image = cv2.resize(image, (new_w, new_h))

        # Now crop to fit the model input size if the resized image is smaller than target
        top = (target_height - new_h) // 2
        bottom = target_height - new_h - top
        left = (target_width - new_w) // 2
        right = target_width - new_w - left

        # Crop the resized image to fit the target size
        cropped_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # Convert to grayscale and normalize
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        normalized = gray.astype(np.float32) / 255.0

        # Add batch and channel dimensions
        return np.expand_dims(np.expand_dims(normalized, axis=0), axis=0), (h, w)


    def postprocess(self, outputs, image):
        """Draw bounding boxes on the image based on model outputs."""
        output_cov = outputs[0]  # Shape: [batch, 5, H, W]
        output_bbox = outputs[1]  # Shape: [batch, 20, H, W]

        height, width, _ = image.shape

        # Process bounding boxes
        bounding_boxes = self.extract_bounding_boxes(output_cov, output_bbox, width, height)
        if not bounding_boxes:
            print("No bounding boxes detected!")  # Debugging
        for box in bounding_boxes:
            x_min, y_min, x_max, y_max = box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        return image

    def extract_bounding_boxes(self, output_cov, output_bbox, img_width, img_height):
        """Extract bounding boxes from model outputs."""
        threshold = 0.2  # Confidence threshold
        bounding_boxes = []

        cov_map = output_cov[0]  # Assuming batch size of 1
        bbox_map = output_bbox[0]  # Assuming batch size of 1

        # Iterate through all grid cells
        for c in range(cov_map.shape[0]):  # Iterate over classes
            for h in range(cov_map.shape[1]):  # Iterate over height
                for w in range(cov_map.shape[2]):  # Iterate over width
                    confidence = cov_map[c, h, w]
                    if confidence > threshold:
                        # Extract bbox coordinates from bbox_map (adjust indices if necessary)
                        x_center = bbox_map[4 * c + 0, h, w] * img_width
                        y_center = bbox_map[4 * c + 1, h, w] * img_height
                        width = bbox_map[4 * c + 2, h, w] * img_width
                        height = bbox_map[4 * c + 3, h, w] * img_height

                        # Calculate bounding box corners
                        x_min = int(x_center - width / 2)
                        y_min = int(y_center - height / 2)
                        x_max = int(x_center + width / 2)
                        y_max = int(y_center + height / 2)

                        # Make sure coordinates are within image boundaries
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        x_max = min(img_width, x_max)
                        y_max = min(img_height, y_max)

                        bounding_boxes.append((x_min, y_min, x_max, y_max))

        print(f"Extracted bounding boxes: {bounding_boxes}")  # Debugging
        return bounding_boxes

    def run_inference(self):
        if self.image is None:
            self.label_result.setText("No image loaded!")
            return

        # Preprocess the image
        self.processed_image, original_shape, pad_info = self.preprocess_image(self.image)

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: self.processed_image})

        # Postprocess and display
        result_image = self.postprocess(outputs, self.image.copy())
        self.show_image(result_image, result=True)
        self.label_result.setText("Inference completed and BBX drawn.")

    def show_image(self, image, result=False):
        """Convert OpenCV image to QPixmap and display."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # Ensure the image will fit within the label's size
        pixmap = pixmap.scaled(self.label_image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Decide which label to use for image display
        if result:
            self.label_result.setPixmap(pixmap)
            self.label_result.setScaledContents(False)  # Prevent scaling to avoid distortion
        else:
            self.label_image.setPixmap(pixmap)
            self.label_image.setScaledContents(False)  # Prevent scaling to avoid distortion


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
