import os
import math
import time

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
LIVE_DURATION_SECONDS = 10
BBOX_PADDING_RATIO = 0.10
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

THUMB_TIP_INDEX = 4
INDEX_FINGER_TIP_INDEX = 8

COLOR_GREEN = (0, 255, 0)
COLOR_ORANGE = (0, 165, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_CYAN = (255, 255, 0)
BOX_THICKNESS = 2

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task"
)
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Results Test"
)

FLASH_DURATION_FRAMES = 8


# ---------------------------------------------------------------------------
# Core Tracking Logic
# ---------------------------------------------------------------------------

def extract_finger_landmarks(hand_landmarks, frame_width, frame_height):
    """Extract pixel coordinates for thumb tip and index finger tip."""
    thumb = hand_landmarks[THUMB_TIP_INDEX]
    index = hand_landmarks[INDEX_FINGER_TIP_INDEX]

    thumb_px = (int(thumb.x * frame_width), int(thumb.y * frame_height))
    index_px = (int(index.x * frame_width), int(index.y * frame_height))

    return thumb_px, index_px


def collect_all_fingertips(all_hand_landmarks, frame_width, frame_height):
    """Gather thumb and index tip pixel coords from all detected hands."""
    points = []
    for hand_landmarks in all_hand_landmarks:
        thumb_px, index_px = extract_finger_landmarks(
            hand_landmarks, frame_width, frame_height
        )
        points.append(thumb_px)
        points.append(index_px)
    return points


def compute_dynamic_polygon(fingertip_points, frame_width, frame_height):
    """Calculate a dynamic polygon (quadrilateral) from fingertip points."""
    if len(fingertip_points) < 3:
        # Fallback to an axis-aligned box for 1 hand (2 points)
        x_coords = [p[0] for p in fingertip_points]
        y_coords = [p[1] for p in fingertip_points]
        
        x_min_raw, x_max_raw = min(x_coords), max(x_coords)
        y_min_raw, y_max_raw = min(y_coords), max(y_coords)

        # Enforce minimum size to avoid flat lines
        if x_max_raw - x_min_raw < 10:
            x_min_raw -= 5; x_max_raw += 5
        if y_max_raw - y_min_raw < 10:
            y_min_raw -= 5; y_max_raw += 5

        pad_x = int((x_max_raw - x_min_raw) * BBOX_PADDING_RATIO)
        pad_y = int((y_max_raw - y_min_raw) * BBOX_PADDING_RATIO)

        x_min = max(0, x_min_raw - pad_x)
        y_min = max(0, y_min_raw - pad_y)
        x_max = min(frame_width, x_max_raw + pad_x)
        y_max = min(frame_height, y_max_raw + pad_y)

        return np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ], dtype=np.int32)
    
    # For 2 hands (4 points), order them cyclically to form a convex polygon
    centroid_x = sum(p[0] for p in fingertip_points) / len(fingertip_points)
    centroid_y = sum(p[1] for p in fingertip_points) / len(fingertip_points)

    def angle_from_centroid(p):
        return math.atan2(p[1] - centroid_y, p[0] - centroid_x)

    ordered_points = sorted(fingertip_points, key=angle_from_centroid)

    # Pad outward from the centroid
    padded_polygon = []
    for p in ordered_points:
        dx = p[0] - centroid_x
        dy = p[1] - centroid_y
        
        # Apply padding ratio
        new_x = int(centroid_x + dx * (1.0 + BBOX_PADDING_RATIO * 1.5)) 
        new_y = int(centroid_y + dy * (1.0 + BBOX_PADDING_RATIO * 1.5))
        
        # Clamp to frame edges
        new_x = max(0, min(frame_width - 1, new_x))
        new_y = max(0, min(frame_height - 1, new_y))
        
        padded_polygon.append([new_x, new_y])
        
    return np.array(padded_polygon, dtype=np.int32)


def is_valid_polygon(polygon):
    """Check that the polygon has a valid area."""
    if polygon is None or len(polygon) < 3:
        return False
    # Use bounding rectangle width/height to quickly check validity
    x, y, w, h = cv2.boundingRect(polygon)
    return w > 0 and h > 0


def create_hand_landmarker():
    """Initialize the MediaPipe HandLandmarker with VIDEO running mode."""
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    return HandLandmarker.create_from_options(options)


def detect_all_hands(landmarker, frame, timestamp_ms):
    """Run hand detection, returning all detected hands or None."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if not result.hand_landmarks:
        return None

    return result.hand_landmarks


# ---------------------------------------------------------------------------
# State visualizers and Capture
# ---------------------------------------------------------------------------

def composite_live_roi(frozen_bg, live_frame, locked_polygon):
    """Paste the live ROI onto the frozen background using a dynamic polygon mask."""
    composite = frozen_bg.copy()
    
    # Create mask for the exact dynamic polygon area
    mask = np.zeros(composite.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [locked_polygon], 255)
    
    # Expand dims for channel broadcasting and apply composite
    mask_bool = mask[:, :, None] == 255
    composite = np.where(mask_bool, live_frame, composite)

    # Draw border around the live ROI to emphasize the portal
    cv2.polylines(
        composite, [locked_polygon], isClosed=True,
        color=COLOR_GREEN, thickness=BOX_THICKNESS
    )

    return composite


def apply_flash_overlay(frame, intensity):
    """Apply a white flash overlay with the given intensity (0.0 to 1.0)."""
    white = np.full_like(frame, 255)
    blended = cv2.addWeighted(frame, 1.0 - intensity, white, intensity, 0)
    return blended


def ensure_results_directory():
    """Create the Results Test folder if it does not exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


def save_capture(frame, capture_number):
    """Save the current frame to the Results Test folder."""
    ensure_results_directory()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"gridcam_{timestamp}_{capture_number:03d}.png"
    filepath = os.path.join(RESULTS_DIR, filename)
    cv2.imwrite(filepath, frame)
    return filepath


def draw_bottom_right_text(frame, text, font=cv2.FONT_HERSHEY_SIMPLEX, scale=3, thickness=4, color=(255, 255, 255)):
    """Draw text on the bottom right of the given frame."""
    frame_height, frame_width = frame.shape[:2]
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    
    # Calculate bottom right position
    margin = 30
    x = frame_width - text_width - margin
    y = frame_height - margin
    
    # Draw outline for better visibility
    cv2.putText(frame, text, (x, y), font, scale, (0, 0, 0), thickness + 2)
    # Draw inner text
    cv2.putText(frame, text, (x, y), font, scale, color, thickness)


# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------

def run_gridcam():
    """Main application loop implementing the 10s LIVE -> FROZEN -> CAPTURE flow."""
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("[ERROR] Could not open camera at index", CAMERA_INDEX)
        print("        Try changing CAMERA_INDEX to 1 or 2.")
        return

    print("[INFO] Camera opened successfully.")
    print("[INFO] Flow: 10s Live Countdown -> Freeze -> SPACE to capture -> Restart")
    
    landmarker = create_hand_landmarker()
    frame_start_ms = int(time.time() * 1000)

    # State Variables
    state = 'LIVE'
    cycle_start = time.time()
    
    last_known_poly = None
    last_known_points = None
    locked_poly = None
    frozen_background = None
    capture_count = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]

            current_ms = int(time.time() * 1000)
            timestamp_ms = current_ms - frame_start_ms

            # ---------------------------------------------------------
            # 1. ALWAYS RUN HAND DETECTION for tracking
            # ---------------------------------------------------------
            all_hands = detect_all_hands(landmarker, frame, timestamp_ms)

            if state == 'LIVE':
                if all_hands is not None:
                    fingertip_points = collect_all_fingertips(
                        all_hands, frame_width, frame_height
                    )
                    poly = compute_dynamic_polygon(
                        fingertip_points, frame_width, frame_height
                    )
                    if is_valid_polygon(poly):
                        last_known_poly = poly
                        last_known_points = fingertip_points
                else:
                    last_known_poly = None
                    last_known_points = None

            # ---------------------------------------------------------
            # 2. STATE LOGIC & RENDERING
            # ---------------------------------------------------------
            display_frame = frame.copy()

            if state == 'LIVE':
                elapsed = time.time() - cycle_start
                remaining = max(0.0, LIVE_DURATION_SECONDS - elapsed)

                # Draw the dynamic grid and finger dots if tracked
                if last_known_poly is not None:
                    cv2.polylines(
                        display_frame, [last_known_poly], isClosed=True,
                        color=COLOR_ORANGE, thickness=BOX_THICKNESS
                    )
                    if last_known_points is not None:
                        for p in last_known_points:
                            cv2.circle(display_frame, p, 5, COLOR_CYAN, -1)
                
                # Bottom right Countdown ONLY in the last 5 seconds
                if remaining <= 5:
                    draw_bottom_right_text(display_frame, f"{int(remaining)+1}")

                if elapsed >= LIVE_DURATION_SECONDS:
                    # Time's up! Transition to FROZEN
                    state = 'FROZEN'
                    frozen_background = frame.copy()
                    
                    if last_known_poly is not None:
                        locked_poly = last_known_poly
                        print(f"[INFO] Freezing! Polygon locked.")
                    else:
                        # Fallback box if no hands were present
                        margin_x = frame_width // 4
                        margin_y = frame_height // 4
                        locked_poly = np.array([
                            [margin_x, margin_y],
                            [frame_width - margin_x, margin_y],
                            [frame_width - margin_x, frame_height - margin_y],
                            [margin_x, frame_height - margin_y]
                        ], dtype=np.int32)
                        print("[WARN] No hands. Using fallback freeze box.")

            elif state == 'FROZEN':
                # Background is frozen, inside locked_poly is live feed
                display_frame = composite_live_roi(
                    frozen_background, frame, locked_poly
                )

            # ---------------------------------------------------------
            # 3. DISPLAY & INPUT LOGIC
            # ---------------------------------------------------------
            cv2.imshow("Gridcam", display_frame)

            key = cv2.waitKey(1) & 0xFF

            # IF IN FROZEN STATE AND SPACE IS PRESSED -> Capture and Reset
            if state == 'FROZEN' and key == 32: # SPACE
                capture_count += 1

                # Save the exactly rendered composite (including the bounding box)
                filepath = save_capture(display_frame, capture_count)
                print(f"[CAPTURE #{capture_count}] Saved: {filepath}")

                # Flash animation over frozen image
                for i in range(FLASH_DURATION_FRAMES, 0, -1):
                    intensity = i / FLASH_DURATION_FRAMES
                    flash_frame = apply_flash_overlay(display_frame, intensity)
                    cv2.imshow("Gridcam", flash_frame)
                    cv2.waitKey(40)

                # Hold briefly after flash
                cv2.imshow("Gridcam", display_frame)
                cv2.waitKey(300)

                # RESTART CYCLE
                print("[INFO] Restarting 10-second cycle.")
                state = 'LIVE'
                cycle_start = time.time()
                last_known_poly = None

            # Quit
            if key == ord('q') or key == 27:
                print("[INFO] Quit signal received. Shutting down.")
                break

    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()
        print(f"[INFO] {capture_count} photos captured. Goodbye.")


if __name__ == "__main__":
    run_gridcam()
