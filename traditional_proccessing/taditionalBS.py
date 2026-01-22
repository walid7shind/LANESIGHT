import math
import numpy as np
import cv2
import os
from pathlib import Path
import argparse

try:
    from color_based import compute_lane_weighted_edges, split_and_score_lines, weighted_fit
except Exception:
    compute_lane_weighted_edges = None
    split_and_score_lines = None
    weighted_fit = None


# =================================================
# Stabilization parameters
# =================================================
MIN_ABS_SLOPE = 0.5
MAX_ABS_SLOPE = 2.5

ALPHA = 0.85
prev_left = None  # (m, c)
prev_right = None  # (m, c)


def reset_smoothing_state():
    global prev_left, prev_right
    prev_left = None
    prev_right = None


def smooth(prev, curr):
    if curr is None:
        return prev
    if prev is None:
        return curr
    return (
        ALPHA * prev[0] + (1 - ALPHA) * curr[0],
        ALPHA * prev[1] + (1 - ALPHA) * curr[1],
    )


def weighted_avg(lines):
    # lines: list of (m, c, length)
    m = np.average([l[0] for l in lines], weights=[l[2] for l in lines])
    c = np.average([l[1] for l in lines], weights=[l[2] for l in lines])
    return float(m), float(c)


def _select_strongest_contributors(
    scored_lines,
    *,
    max_keep: int = 8,
    min_keep: int = 2,
    min_len_ratio: float = 0.35,
    cum_len_ratio: float = 0.75,
):
    """Pick the longest, most meaningful segments.

    scored_lines: list of (slope, intercept, length)
    """
    if not scored_lines:
        return []

    scored_sorted = sorted(scored_lines, key=lambda t: t[2], reverse=True)
    max_len = float(scored_sorted[0][2])
    if max_len <= 1e-6:
        return []

    filtered = [t for t in scored_sorted if float(t[2]) >= (min_len_ratio * max_len)]
    if len(filtered) < min_keep:
        filtered = scored_sorted

    filtered = filtered[: max_keep]

    total_len = float(sum(t[2] for t in filtered))
    if total_len <= 1e-6:
        return []

    keep = []
    acc = 0.0
    for t in filtered:
        keep.append(t)
        acc += float(t[2])
        if len(keep) >= max_keep:
            break
        if len(keep) >= min_keep and (acc / total_len) >= cum_len_ratio:
            break

    return keep

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    # Expected input is BGR (cv2 default)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    kernel_dilate = cv2.getStructuringElement(
    cv2.MORPH_RECT, (2, 4)
)
    kernel_erode = cv2.getStructuringElement(
    cv2.MORPH_RECT, (1, 1)
)       
    kernel_dilate = cv2.getStructuringElement(
    cv2.MORPH_RECT, (2, 8)
)

    

    img= cv2.Canny(img, low_threshold, high_threshold)
    img = cv2.erode(img, kernel_erode, iterations=1)
    img = cv2.dilate(img, kernel_dilate, iterations=1)
    
    return img

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def slope_lines(image, lines):
    global prev_left, prev_right
    if lines is None:
        return None
    img = image.copy()
    poly_vertices = []
    order = [0, 1, 3, 2]

    if split_and_score_lines is None or weighted_fit is None:
        # Fallback to previous behavior if helpers are unavailable
        left_lines = []   # (m, c, length)
        right_lines = []  # (m, c, length)
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1 == x2:
                    continue
                m = (y2 - y1) / (x2 - x1)
                c = y1 - m * x1
                if abs(m) < MIN_ABS_SLOPE or abs(m) > MAX_ABS_SLOPE:
                    continue
                seg_len = float(np.hypot(x2 - x1, y2 - y1))
                if seg_len <= 1e-6:
                    continue
                if m < 0:
                    left_lines.append((m, c, seg_len))
                else:
                    right_lines.append((m, c, seg_len))
        curr_left = weighted_avg(left_lines) if len(left_lines) > 0 else None
        curr_right = weighted_avg(right_lines) if len(right_lines) > 0 else None
    else:
        # Required behavior:
        # per side: filter by slope -> rank by length -> keep strongest -> fit ONE line
        left_scored, right_scored = split_and_score_lines(lines, min_slope=MIN_ABS_SLOPE)

        left_scored = [t for t in left_scored if abs(t[0]) <= MAX_ABS_SLOPE]
        right_scored = [t for t in right_scored if abs(t[0]) <= MAX_ABS_SLOPE]

        left_best = _select_strongest_contributors(left_scored)
        right_best = _select_strongest_contributors(right_scored)

        curr_left = weighted_fit(left_best)
        curr_right = weighted_fit(right_best)

    prev_left = smooth(prev_left, curr_left)
    prev_right = smooth(prev_right, curr_right)

    if prev_left is None or prev_right is None:
        # Not enough signal yet
        return None

    left_line = prev_left
    right_line = prev_right

    for slope, intercept in (left_line, right_line):
        rows, cols = image.shape[:2]
        y1 = rows
        y2 = int(rows * 0.6)

        if abs(slope) < 1e-6:
            continue

        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

        poly_vertices.append((x1, y1))
        poly_vertices.append((x2, y2))

        draw_lines(img, np.array([[[x1, y1, x2, y2]]]))

    if len(poly_vertices) < 4:
        return None

    poly_vertices = [poly_vertices[i] for i in order]
    cv2.fillPoly(img, pts=np.array([poly_vertices], 'int32'), color=(255, 255, 255))
    return poly_vertices

    
    #cv2.polylines(img,np.array([poly_vertices],'int32'), True, (0,0,255), 10)
    #print(poly_vertices)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
 
    lines = cv2.HoughLinesP(
        img,
        rho,
        theta,
        threshold,
        np.array([]),
        minLineLength=min_line_len,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return None

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    poly = slope_lines(line_img, lines)
    return poly


def polygon_to_mask(image_shape, poly_vertices):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if poly_vertices is None:
        return mask
    cv2.fillPoly(mask, [np.array(poly_vertices, dtype=np.int32)], 1)
    return mask


def traditional_polygon_mask(image_bgr):
    gray = grayscale(image_bgr)
    blur = gaussian_blur(gray, 5)
    edges = canny(blur, 180, 240)
    masked = region_of_interest(edges, get_vertices(image_bgr))

    poly = hough_lines(
        img=masked,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        min_line_len=20,
        max_line_gap=180,
    )

    return polygon_to_mask(image_bgr.shape, poly)

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.1, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    lines_edges = cv2.addWeighted(initial_img, α, img, β, γ)
    #lines_edges = cv2.polylines(lines_edges,get_vertices(img), True, (0,0,255), 10)
    return lines_edges
def get_vertices(image):
    h, w = image.shape[:2]

    bottom_left  = (0, h)
    top_left     = (0, int(0.7 * h))
    top_right    = (w, int(0.7 * h))
    bottom_right = (w, h)

    return np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

    
    ver = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return ver
# Lane finding Pipeline
def lane_finding_pipeline(image):
    return traditional_polygon_mask(image)


def lane_finding_pipeline_debug(image_bgr: np.ndarray, use_color_weighting: bool = True):
    gray_img = grayscale(image_bgr)
    smoothed_img = gaussian_blur(img=gray_img, kernel_size=5)
    canny_img = canny(img=smoothed_img, low_threshold=180, high_threshold=240)
    canny_for_roi = canny_img
    if use_color_weighting and compute_lane_weighted_edges is not None:
        # Optional (used with caution): suppress road-like edges, emphasize lane-like colors
        canny_for_roi = compute_lane_weighted_edges(image_bgr, canny_img)

    masked_img = region_of_interest(img=canny_for_roi, vertices=get_vertices(image_bgr))
    poly = hough_lines(
        img=masked_img,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        min_line_len=20,
        max_line_gap=180,
    )
    mask = polygon_to_mask(image_bgr.shape, poly)
    return mask, canny_for_roi, masked_img, poly

