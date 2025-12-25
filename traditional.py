import cv2
import numpy as np

# 1. Load image
image = cv2.imread("road.jpg")
if image is None:
    raise FileNotFoundError("Image 'road.jpg' not found in current directory!")

# 2. Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. Gaussian Blur (reduce noise)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 4. Canny Edge Detection
edges = cv2.Canny(blur, 50, 150)
cv2.imshow("canny", edges)
# 5. Define Region of Interest (ROI)
height, width = edges.shape
mask = np.zeros_like(edges)

# polygon for road area (trapezoid)
roi_vertices = np.array([[
    (50, height),
    (width // 2 - 50, height // 2 + 50),
    (width // 2 + 50, height // 2 + 50),
    (width - 50, height)
]], dtype=np.int32)

cv2.fillPoly(mask, roi_vertices, 255)
masked_edges = cv2.bitwise_and(edges, mask)

# 6. Hough Transform for line detection
lines = cv2.HoughLinesP(masked_edges,
                        rho=1,
                        theta=np.pi/180,
                        threshold=50,
                        minLineLength=50,
                        maxLineGap=150)

# 7. Draw lines on a copy of the image
line_image = np.copy(image)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

# 8. Overlay detected lines on original image
output = cv2.addWeighted(image, 0.8, line_image, 1, 0)

# 9. Display result
cv2.imshow("Detected Lanes", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
