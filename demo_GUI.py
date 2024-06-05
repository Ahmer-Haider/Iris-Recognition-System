import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QPushButton, QVBoxLayout, QWidget, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import numpy as np
import pandas as pd
import math
import time

def load_image(filepath):
    img = cv2.imread(filepath, 0)
    return img

def get_iris_boundaries(img):
    pupil_circle = find_pupil(img)
    if not pupil_circle:
        return None, None

    radius_range = int(math.ceil(pupil_circle[2] * 1.5))
    multiplier = 0.25
    center_range = int(math.ceil(pupil_circle[2] * multiplier))
    ext_iris_circle = find_ext_iris(img, pupil_circle, center_range, radius_range)

    while not ext_iris_circle and multiplier <= 0.7:
        multiplier += 0.05
        center_range = int(math.ceil(pupil_circle[2] * multiplier))
        ext_iris_circle = find_ext_iris(img, pupil_circle, center_range, radius_range)

    if not ext_iris_circle:
        return None, None

    return pupil_circle, ext_iris_circle

def find_pupil(img):
    def get_edges(image):
        edges = cv2.Canny(image, 20, 100)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        ksize = 2 * np.random.randint(5, 11) + 1
        edges = cv2.GaussianBlur(edges, (ksize, ksize), 0)
        return edges

    param1 = 200
    param2 = 120
    pupil_circles = []
    while param2 > 35 and len(pupil_circles) < 100:
        for mdn, thrs in [(m, t) for m in [3, 5, 7] for t in range(20, 61, 5)]:
            median = cv2.medianBlur(img, 2 * mdn + 1)
            ret, thres = cv2.threshold(median, thrs, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thres.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            thres = cv2.drawContours(thres, contours, -1, (255), -1)
            edges = get_edges(thres)
            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 1, np.array([]), param1, param2)
            if circles is not None and len(circles) > 0:
                circles = np.round(circles[0, :]).astype("int")
                pupil_circles.extend(circles)
        param2 -= 1

    return get_mean_circle(pupil_circles)

def get_mean_circle(circles):
    if not circles:
        return None
    mean_x = int(np.mean([c[0] for c in circles]))
    mean_y = int(np.mean([c[1] for c in circles]))
    mean_r = int(np.mean([c[2] for c in circles]))
    return mean_x, mean_y, mean_r

def find_ext_iris(img, pupil_circle, center_range, radius_range):
    def get_edges(image, thrs2):
        thrs1 = 0
        edges = cv2.Canny(image, thrs1, thrs2, apertureSize=5)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        ksize = 2 * np.random.randint(5, 11) + 1
        edges = cv2.GaussianBlur(edges, (ksize, ksize), 0)
        return edges

    def get_circles(hough_param, median_params, edge_params):
        crt_circles = []
        for mdn, thrs2 in [(m, t) for m in median_params for t in edge_params]:
            median = cv2.medianBlur(img, 2 * mdn + 1)
            edges = get_edges(median, thrs2)
            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 1, np.array([]), 200, hough_param)
            if circles is not None and len(circles) > 0:
                circles = np.round(circles[0, :]).astype("int")
                for (c_col, c_row, r) in circles:
                    if point_in_circle(pupil_circle[0], pupil_circle[1], center_range, c_col, c_row) and r > radius_range:
                        crt_circles.append((c_col, c_row, r))
        return crt_circles

    param2 = 120
    total_circles = []
    while param2 > 40 and len(total_circles) < 50:
        crt_circles = get_circles(param2, [8, 10, 12, 14, 16, 18, 20], [430, 480, 530])
        if crt_circles:
            total_circles.extend(crt_circles)
        param2 -= 1

    if not total_circles:
        return None

    return get_mean_circle(total_circles)

def point_in_circle(x0, y0, r, x, y):
    return (x - x0) ** 2 + (y - y0) ** 2 <= r ** 2

def get_equalized_iris(img, ext_iris_circle, pupil_circle):
    x_pupil, y_pupil, r_pupil = pupil_circle
    x_iris, y_iris, r_iris = ext_iris_circle

    theta = np.linspace(0, 2 * np.pi, 360)
    b = np.linspace(r_pupil, r_iris, r_iris - r_pupil)

    pupil_center = np.array([x_pupil, y_pupil])

    yi = pupil_center[1] + np.round(np.outer(b, np.sin(theta))).astype(int)
    xi = pupil_center[0] + np.round(np.outer(b, np.cos(theta))).astype(int)

    roi = img[yi, xi]
    return roi

def extract_features(roi):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(roi, None)
    return keypoints, descriptors

def match_features(des1, des2, ratio=0.75):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches

def read_image_paths_from_excel(file_path, column_name='IRIS PATH'):
    try:
        df = pd.read_excel(file_path)
        image_paths = df[column_name].tolist()
        return image_paths, df
    except Exception as e:
        print(f"An error occurred while reading the Excel file: {e}")
        return [], None

class IrisMatcher(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Iris Matcher")
        self.setGeometry(100, 100, 800, 600)

        self.excel_file_path = None
        self.user_image_path = None

        self.init_ui()

    def init_ui(self):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.load_excel_button = QPushButton("Load Excel File", self.central_widget)
        self.load_excel_button.clicked.connect(self.load_excel_file)
        self.layout.addWidget(self.load_excel_button)

        self.load_image_button = QPushButton("Load Image to Compare", self.central_widget)
        self.load_image_button.clicked.connect(self.load_image_file)
        self.layout.addWidget(self.load_image_button)

        self.image_label = QLabel("Image will be displayed here", self.central_widget)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        self.match_button = QPushButton("Match Iris", self.central_widget)
        self.match_button.clicked.connect(self.match_iris)
        self.layout.addWidget(self.match_button)

        self.result_label = QLabel("", self.central_widget)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.result_label)

    def load_excel_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Excel File", "", "Excel Files (*.xlsx);;All Files (*)", options=options)
        if file_path:
            self.excel_file_path = file_path
            self.result_label.setText(f"Loaded Excel File: {file_path}")

    def load_image_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp *.jpeg);;All Files (*)", options=options)
        if file_path:
            self.user_image_path = file_path
            self.display_image(file_path)
            self.result_label.setText(f"Loaded Image File: {file_path}")

    def display_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap(q_image)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))

    def match_iris(self):
        if not self.excel_file_path or not self.user_image_path:
            QMessageBox.warning(self, "Error", "Please load both the Excel file and the image file.")
            return

        image_paths, df = read_image_paths_from_excel(self.excel_file_path)

        if not image_paths:
            QMessageBox.warning(self, "Error", "No image paths found in the Excel file.")
            return

        img2 = load_image(self.user_image_path)

        if img2 is None:
            QMessageBox.warning(self, "Error", "Error loading user entered image.")
            return

        for idx, image_path in enumerate(image_paths):
            img1 = load_image(image_path)
            if img1 is None:
                print(f"Error loading image at path: {image_path}")
                continue

            pupil_circle1, ext_iris_circle1 = get_iris_boundaries(img1)
            pupil_circle2, ext_iris_circle2 = get_iris_boundaries(img2)

            if pupil_circle1 and ext_iris_circle1 and pupil_circle2 and ext_iris_circle2:
                roi1 = get_equalized_iris(img1, ext_iris_circle1, pupil_circle1)
                roi2 = get_equalized_iris(img2, ext_iris_circle2, pupil_circle2)

                _, des1 = extract_features(roi1)
                _, des2 = extract_features(roi2)

                matches = match_features(des1, des2)
                print(f"Number of good matches for image {idx + 1}: {len(matches)}")

                if len(matches) > 10:
                    self.result_label.setText(f"The irises match for image {idx + 1}.")
                    df.loc[idx, 'CASTED VOTE'] = 1
                    break
                else:
                    self.result_label.setText(f"The irises do not match for image {idx + 1}.")
                    df.loc[idx, 'CASTED VOTE'] = 0
            else:
                self.result_label.setText(f"Failed to detect iris boundaries for image {idx + 1}.")
                df.loc[idx, 'CASTED VOTE'] = 0

        attempts = len(image_paths)
        for attempt in range(attempts):
            try:
                df.to_excel(self.excel_file_path, index=False)
                self.result_label.setText("Data saved successfully.")
                break
            except PermissionError:
                print(f"Attempt {attempt + 1}/{attempts}: Permission denied. Retrying in 2 seconds...")
                time.sleep(2)
        else:
            self.result_label.setText("Failed to save the file after multiple attempts.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = IrisMatcher()
    window.show()
    sys.exit(app.exec_())
