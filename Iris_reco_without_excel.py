import cv2
import numpy as np
import math
import pandas as pd

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
        # Read the Excel file
        df = pd.read_excel(file_path)
        # Extract the column with image paths
        image_paths = df[column_name].tolist()
        return image_paths
    except Exception as e:
        print(f"An error occurred while reading the Excel file: {e}")
        return []

if __name__ == "__main__":
    # Load two iris images
    excel_file_path = 'excelDataSet.xlsx'  # Path to your Excel file
    # user_entered_path = input("Enter the image path to compare: ")

    # Read image paths from the Excel file
    image_paths = read_image_paths_from_excel(excel_file_path)  
    image_path1 = 'S2001R01.jpg'
    image_path2 = 'S2001R01 copy.jpg'

    img1 = load_image(image_path1)
    img2 = load_image(image_path2)

    # Get iris boundaries and equalized iris region for both images
    pupil_circle1, ext_iris_circle1 = get_iris_boundaries(img1)
    pupil_circle2, ext_iris_circle2 = get_iris_boundaries(img2)

    if pupil_circle1 and ext_iris_circle1 and pupil_circle2 and ext_iris_circle2:
        roi1 = get_equalized_iris(img1, ext_iris_circle1, pupil_circle1)
        roi2 = get_equalized_iris(img2, ext_iris_circle2, pupil_circle2)

        # Extract features from both equalized iris regions
        _, des1 = extract_features(roi1)
        _, des2 = extract_features(roi2)

        # Match features
        matches = match_features(des1, des2)
        print(f"Number of good matches: {len(matches)}")

        # Arbitrary threshold to determine a match
        if len(matches) > 10:
            print("The irises match.")
        else:
            print("The irises do not match.")
    else:
        print("Failed to detect iris boundaries in one or both images.")