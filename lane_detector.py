import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Helper Function: Region of Interest
def region_of_interest(img):
    height, width = img.shape

    mask = np.zeros_like(img)

    # Define trapezoid region for lanes
    polygon = np.array([[
        (int(width * 0.15), height),
        (int(width * 0.85), height),
        (int(width * 0.55), int(height * 0.55)),
        (int(width * 0.45), int(height * 0.55))
    ]], np.int32)

    # Fill the region with white
    cv2.fillPoly(mask, polygon, 255)

    # Apply mask
    masked = cv2.bitwise_and(img, mask)

    return masked


# Helper Function: Draw Lane Lines
def draw_lines(img, lines):
    line_img = np.zeros_like(img)

    left_lines = []
    right_lines = []

    if lines is None:
        return img

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Calculate slope
        if x2 - x1 == 0:
            continue

        slope = (y2 - y1) / (x2 - x1)

        # Filter nearly horizontal lines
        if abs(slope) < 0.3:
            continue

        # Separate left and right lanes
        if slope < 0:
            left_lines.append(line[0])
        else:
            right_lines.append(line[0])

    def average_line(lines):
        if len(lines) == 0:
            return None

        x_coords = []
        y_coords = []

        for x1, y1, x2, y2 in lines:
            x_coords += [x1, x2]
            y_coords += [y1, y2]

        # Fit a single line using polyfit
        fit = np.polyfit(x_coords, y_coords, 1)
        slope, intercept = fit

        y1 = img.shape[0]
        y2 = int(img.shape[0] * 0.6)

        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

        return x1, y1, x2, y2

    left_avg = average_line(left_lines)
    right_avg = average_line(right_lines)

    if left_avg is not None:
        cv2.line(line_img, (left_avg[0], left_avg[1]),
                 (left_avg[2], left_avg[3]), (0, 0, 255), 5)

    if right_avg is not None:
        cv2.line(line_img, (right_avg[0], right_avg[1]),
                 (right_avg[2], right_avg[3]), (0, 0, 255), 5)

    # Combine original image with lane lines
    result = cv2.addWeighted(img, 0.8, line_img, 1, 0)

    return result

# Main Function for Image
def process_image(image):

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Apply adaptive threshold to detect edges in varying lighting conditions
    adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Combine Canny edges and Adaptive Threshold
    combined = cv2.bitwise_or(edges, adaptive)

    # Apply Region of Interest mask
    roi = region_of_interest(combined)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(
        roi,
        1,
        np.pi / 180,
        threshold=30,
        minLineLength=20,
        maxLineGap=200
    )

    # Draw detected lane lines
    result = draw_lines(image, lines)

    return result

# 6-Panel Visualization for Image Output
def show_pipeline(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    combined = cv2.bitwise_or(edges, adaptive)
    roi = region_of_interest(combined)
    detected = process_image(image)

    fig, axes = plt.subplots(3, 2, figsize=(10, 10))

    axes[0, 0].imshow(edges, cmap='gray')
    axes[0, 0].set_title('Edges')

    axes[0, 1].imshow(cv2.cvtColor(detected, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Detected Path')

    axes[1, 0].imshow(adaptive, cmap='gray')
    axes[1, 0].set_title('Adaptive Thresh')

    axes[1, 1].imshow(roi, cmap='gray')
    axes[1, 1].set_title('ROI')

    axes[2, 0].imshow(gray, cmap='gray')
    axes[2, 0].set_title('Gray')

    axes[2, 1].imshow(combined, cmap='gray')
    axes[2, 1].set_title('Combined')

    plt.tight_layout()
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/output_pipeline.png')
    plt.show()

# Process a video and save output
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = process_image(frame)
        out.write(result)

        cv2.imshow("Lane Detection", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Saved: {output_path}")


# 
os.makedirs('output', exist_ok=True)

# 1. Process image
image = cv2.imread("input/image.jpg")
if image is not None:
    result = process_image(image)
    cv2.imwrite('output/output_detected.png', result)
    print("Saved: output/output_detected.png")
    show_pipeline(image)
else:
    print("WARNING: input/image.jpg not found, skipping")

# 2. Process input/road.mp4
if os.path.exists("input/road.mp4"):
    process_video("input/road.mp4", "output/output_road.mp4")
else:
    print("WARNING: input/road.mp4 not found, skipping")

# 3. Process input/test_video.mp4
if os.path.exists("input/test_video.mp4"):
    process_video("input/test_video.mp4", "output/output_test_video.mp4")
else:
    print("WARNING: input/test_video.mp4 not found, skipping")