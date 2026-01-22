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
        return image
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
        return image

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

    poly_vertices = [poly_vertices[i] for i in order]
    cv2.fillPoly(img, pts=np.array([poly_vertices], 'int32'), color=(0, 255, 0))

    return cv2.addWeighted(image, 0.7, img, 0.4, 0.0)

    
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
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #draw_lines(line_img, lines)
    if lines is None:
        return line_img

    line_img = slope_lines(line_img, lines)
    return line_img

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
    
    #Grayscale
    gray_img = grayscale(image)
    #Gaussian Smoothing
    smoothed_img = gaussian_blur(img = gray_img, kernel_size = 5)
    #Canny Edge Detection
    canny_img = canny(img = smoothed_img, low_threshold = 180, high_threshold = 240)
    #Masked Image Within a Polygon
    masked_img = region_of_interest(img = canny_img, vertices = get_vertices(image))
    #Hough Transform Lines
    houghed_lines = hough_lines(img = masked_img, rho = 1, theta = np.pi/180, threshold = 20, min_line_len = 20, max_line_gap = 180)
    #Draw lines on edges
    output = weighted_img(img = houghed_lines, initial_img = image, α=0.8, β=1., γ=0.)
    
    return output


def lane_finding_pipeline_debug(image_bgr: np.ndarray, use_color_weighting: bool = True):
    gray_img = grayscale(image_bgr)
    smoothed_img = gaussian_blur(img=gray_img, kernel_size=5)
    canny_img = canny(img=smoothed_img, low_threshold=180, high_threshold=240)
    canny_for_roi = canny_img
    if use_color_weighting and compute_lane_weighted_edges is not None:
        # Optional (used with caution): suppress road-like edges, emphasize lane-like colors
        canny_for_roi = compute_lane_weighted_edges(image_bgr, canny_img)

    masked_img = region_of_interest(img=canny_for_roi, vertices=get_vertices(image_bgr))
    houghed_lines = hough_lines(
        img=masked_img,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        min_line_len=20,
        max_line_gap=180,
    )
    output = weighted_img(img=houghed_lines, initial_img=image_bgr, α=0.8, β=1.0, γ=0.0)
    return output, canny_for_roi, masked_img, houghed_lines


def _to_bgr_3ch(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _resize_keep_aspect(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(
        out,
        text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def make_debug_montage(canny_img, roi_img, hough_img, weighted_img_bgr, tile_w=640, tile_h=360):
    canny_bgr = _label(_resize_keep_aspect(_to_bgr_3ch(canny_img), tile_w, tile_h), "Canny")
    roi_bgr = _label(_resize_keep_aspect(_to_bgr_3ch(roi_img), tile_w, tile_h), "Region of interest")
    hough_bgr = _label(_resize_keep_aspect(_to_bgr_3ch(hough_img), tile_w, tile_h), "Hough lines")
    weighted_bgr = _label(_resize_keep_aspect(_to_bgr_3ch(weighted_img_bgr), tile_w, tile_h), "Weighted")

    top = cv2.hconcat([canny_bgr, roi_bgr])
    bottom = cv2.hconcat([hough_bgr, weighted_bgr])
    return cv2.vconcat([top, bottom])


def _iter_inputs(input_path: Path):
    if input_path.is_file():
        yield input_path
        return

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}

    files = sorted([p for p in input_path.iterdir() if p.is_file()])
    image_files = [p for p in files if p.suffix.lower() in image_exts]
    if image_files:
        for p in image_files:
            yield p
        return

    video_files = [p for p in files if p.suffix.lower() in video_exts]
    if video_files:
        yield video_files[0]
        return

    raise FileNotFoundError(f"No supported images/videos found in: {input_path}")


def _is_video(path: Path) -> bool:
    return path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}


def main():
    parser = argparse.ArgumentParser(description="Traditional lane detection (Canny + ROI + Hough) with debug display")
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "test_images"),
        help="Path to an image, a video, or a folder (default: ./test_images)",
    )
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame for video")
    parser.add_argument("--tile-w", type=int, default=640, help="Debug tile width")
    parser.add_argument("--tile-h", type=int, default=360, help="Debug tile height")
    parser.add_argument(
        "--use-color-weighting",
        action="store_true",
        help="Use Lab-based color weighting from color_based.py (optional)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    window_name = "Lane debug (q=quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for path in _iter_inputs(input_path):
        if _is_video(path):
            reset_smoothing_state()
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {path}")

            frame_idx = 0
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                if args.stride > 1 and (frame_idx % args.stride != 0):
                    frame_idx += 1
                    continue

                weighted_out, canny_img, roi_img, hough_img = lane_finding_pipeline_debug(
                    frame_bgr,
                    use_color_weighting=args.use_color_weighting,
                )
                montage = make_debug_montage(
                    canny_img,
                    roi_img,
                    hough_img,
                    weighted_out,
                    tile_w=args.tile_w,
                    tile_h=args.tile_h,
                )
                cv2.imshow(window_name, montage)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

                frame_idx += 1

            cap.release()
        else:
            reset_smoothing_state()
            img_bgr = cv2.imread(str(path))
            if img_bgr is None:
                print(f"Skipping unreadable file: {path}")
                continue

            weighted_out, canny_img, roi_img, hough_img = lane_finding_pipeline_debug(
                img_bgr,
                use_color_weighting=args.use_color_weighting,
            )
            montage = make_debug_montage(
                canny_img,
                roi_img,
                hough_img,
                weighted_out,
                tile_w=args.tile_w,
                tile_h=args.tile_h,
            )
            cv2.imshow(window_name, montage)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q") or key == 27:
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()