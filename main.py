"""
Project Gridcam — 30-Second Cyclic Camera State Machine

Tracks thumb tip (Landmark 4) and index finger tip (Landmark 8) from
BOTH hands using MediaPipe Hands (Tasks API). Uses all detected
fingertips to define a bounding box, allowing a larger grid when
both hands are visible. Composites a live ROI onto a frozen
background in a 30-second cycle.
"""

import os
import time

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
CYCLE_DURATION_SECONDS = 30
TRANSITION_TIME_SECONDS = 15
BBOX_PADDING_RATIO = 0.10
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

THUMB_TIP_INDEX = 4
INDEX_FINGER_TIP_INDEX = 8

COLOR_GREEN = (0, 255, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_CYAN = (0, 255, 255)
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 1

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")


# ---------------------------------------------------------------------------
# Phase 2: Core Tracking Logic
# ---------------------------------------------------------------------------

def extract_finger_landmarks(hand_landmarks, frame_width, frame_height):
    """Extract pixel coordinates for thumb tip and index finger tip.

    Args:
        hand_landmarks: List of NormalizedLandmark from MediaPipe Tasks API.
        frame_width: Width of the video frame in pixels.
        frame_height: Height of the video frame in pixels.

    Returns:
        tuple: ((thumb_x, thumb_y), (index_x, index_y)) in pixel coords.
    """
    thumb = hand_landmarks[THUMB_TIP_INDEX]
    index = hand_landmarks[INDEX_FINGER_TIP_INDEX]

    thumb_px = (int(thumb.x * frame_width), int(thumb.y * frame_height))
    index_px = (int(index.x * frame_width), int(index.y * frame_height))

    return thumb_px, index_px


def collect_all_fingertips(all_hand_landmarks, frame_width, frame_height):
    """Gather thumb + index tip pixel coords from ALL detected hands.

    Supports 1 or 2 hands. With 2 hands, the bounding box spans a
    much larger area — each hand acts as a corner of the grid.

    Returns:
        list: All (x, y) pixel coordinate tuples for detected fingertips.
    """
    points = []
    for hand_landmarks in all_hand_landmarks:
        thumb_px, index_px = extract_finger_landmarks(
            hand_landmarks, frame_width, frame_height
        )
        points.append(thumb_px)
        points.append(index_px)
    return points


def compute_padded_bounding_box(fingertip_points, frame_width, frame_height):
    """Calculate a bounding box around ALL fingertip points with 10% padding.

    Works with 2 points (1 hand) or 4 points (2 hands) for a larger grid.
    The box is clamped to frame dimensions to prevent out-of-bounds slicing.

    Returns:
        tuple: (x_min, y_min, x_max, y_max) in pixel coordinates.
    """
    x_coords = [p[0] for p in fingertip_points]
    y_coords = [p[1] for p in fingertip_points]

    x_min_raw = min(x_coords)
    x_max_raw = max(x_coords)
    y_min_raw = min(y_coords)
    y_max_raw = max(y_coords)

    box_width = x_max_raw - x_min_raw
    box_height = y_max_raw - y_min_raw

    pad_x = int(box_width * BBOX_PADDING_RATIO)
    pad_y = int(box_height * BBOX_PADDING_RATIO)

    # Apply padding and clamp to frame edges
    x_min = max(0, x_min_raw - pad_x)
    y_min = max(0, y_min_raw - pad_y)
    x_max = min(frame_width, x_max_raw + pad_x)
    y_max = min(frame_height, y_max_raw + pad_y)

    return x_min, y_min, x_max, y_max


def is_valid_bounding_box(bbox):
    """Check that the bounding box has non-zero area.

    Prevents empty array slices that would cause compositing failures.
    """
    x_min, y_min, x_max, y_max = bbox
    return (x_max - x_min) > 0 and (y_max - y_min) > 0


def create_hand_landmarker():
    """Initialize the MediaPipe HandLandmarker with VIDEO running mode.

    Returns:
        HandLandmarker instance configured for video frame processing.
    """
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
    """Run hand detection on a single frame, returning ALL detected hands.

    Args:
        landmarker: MediaPipe HandLandmarker instance.
        frame: BGR frame from OpenCV (will be converted to RGB).
        timestamp_ms: Frame timestamp in milliseconds.

    Returns:
        List of hand landmark lists (one per hand), or None if no hands.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if not result.hand_landmarks:
        return None

    return result.hand_landmarks


# ---------------------------------------------------------------------------
# Phase 2: Test Loop — Verify tracking works with live camera
# ---------------------------------------------------------------------------

def run_tracking_test():
    """Open camera, track fingers from both hands, draw bounding box.

    Uses 1 or 2 hands — with both hands the grid spans much larger.
    Press 'q' or ESC to quit.
    """
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("[ERROR] Could not open camera at index", CAMERA_INDEX)
        print("        Try changing CAMERA_INDEX to 1 or 2.")
        return

    print("[INFO] Camera opened successfully.")
    print("[INFO] Phase 2 — Dual-Hand Tracking Test")
    print("[INFO] Show one or BOTH hands to create a grid. Press 'q' or ESC to quit.")

    landmarker = create_hand_landmarker()
    start_time_ms = int(time.time() * 1000)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("[WARN] Failed to read frame, skipping...")
                continue

            frame = cv2.flip(frame, 1)  # Mirror for natural interaction
            frame_height, frame_width = frame.shape[:2]

            # Calculate timestamp relative to start
            current_time_ms = int(time.time() * 1000)
            timestamp_ms = current_time_ms - start_time_ms

            all_hands = detect_all_hands(landmarker, frame, timestamp_ms)

            if all_hands is not None:
                # Collect all fingertip points from all detected hands
                fingertip_points = collect_all_fingertips(
                    all_hands, frame_width, frame_height
                )

                bbox = compute_padded_bounding_box(
                    fingertip_points, frame_width, frame_height
                )

                if is_valid_bounding_box(bbox):
                    x_min, y_min, x_max, y_max = bbox

                    # Draw bounding box
                    cv2.rectangle(
                        frame, (x_min, y_min), (x_max, y_max),
                        COLOR_GREEN, BOX_THICKNESS
                    )

                    # Draw all fingertip points
                    for point in fingertip_points:
                        cv2.circle(frame, point, 5, COLOR_CYAN, -1)

                    # Label with hand count
                    hand_count = len(all_hands)
                    label = f"{hand_count}H | Box: ({x_min},{y_min})-({x_max},{y_max})"
                    cv2.putText(
                        frame, label, (x_min, y_min - 10),
                        FONT, FONT_SCALE, COLOR_GREEN, FONT_THICKNESS
                    )
            else:
                cv2.putText(
                    frame, "No hands detected — show 1 or 2 hands", (10, 30),
                    FONT, FONT_SCALE, COLOR_WHITE, FONT_THICKNESS
                )

            cv2.imshow("Gridcam - Phase 2 Tracking Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("[INFO] Quit signal received. Shutting down.")
                break

    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Resources released. Goodbye.")


if __name__ == "__main__":
    run_tracking_test()
