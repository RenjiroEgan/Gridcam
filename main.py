"""
Project Gridcam -- Photo Capture with Dual-Hand Tracking & PiP

Cycle Flow:
1. Live Tracking (0-10s): Normal camera feed, tracks both hands to draw a clean grid box.
   Displays a countdown in the center for the final 5 seconds.
2. Freeze State (10s+): Background freezes. Bounding box coordinates lock.
   Inside the box is a live video feed (PiP).
3. Capture (SPACE): Pressing SPACE during the freeze state captures the photo,
   applies a flash animation, saves it to 'Results Test', and restarts the cycle.
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
LIVE_DURATION_SECONDS = 10
BBOX_PADDING_RATIO = 0.10
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

THUMB_TIP_INDEX = 4
INDEX_FINGER_TIP_INDEX = 8

COLOR_GREEN = (0, 255, 0)
COLOR_ORANGE = (0, 165, 255)
COLOR_WHITE = (255, 255, 255)
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


def compute_padded_bounding_box(fingertip_points, frame_width, frame_height):
    """Calculate a bounding box with 10% padding, clamped to frame edges."""
    x_coords = [p[0] for p in fingertip_points]
    y_coords = [p[1] for p in fingertip_points]

    x_min_raw, x_max_raw = min(x_coords), max(x_coords)
    y_min_raw, y_max_raw = min(y_coords), max(y_coords)

    pad_x = int((x_max_raw - x_min_raw) * BBOX_PADDING_RATIO)
    pad_y = int((y_max_raw - y_min_raw) * BBOX_PADDING_RATIO)

    x_min = max(0, x_min_raw - pad_x)
    y_min = max(0, y_min_raw - pad_y)
    x_max = min(frame_width, x_max_raw + pad_x)
    y_max = min(frame_height, y_max_raw + pad_y)

    return x_min, y_min, x_max, y_max


def is_valid_bounding_box(bbox):
    """Check that the bounding box has non-zero area."""
    x_min, y_min, x_max, y_max = bbox
    return (x_max - x_min) > 0 and (y_max - y_min) > 0


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

def composite_live_roi(frozen_bg, live_frame, locked_bbox):
    """Paste the live ROI onto the frozen background."""
    x_min, y_min, x_max, y_max = locked_bbox
    composite = frozen_bg.copy()
    
    # Check bounds again just to be 100% safe to prevent slicing errors
    y_max = min(y_max, frozen_bg.shape[0])
    x_max = min(x_max, frozen_bg.shape[1])
    
    composite[y_min:y_max, x_min:x_max] = live_frame[y_min:y_max, x_min:x_max]

    # Draw border around the live ROI to emphasize it
    cv2.rectangle(
        composite, (x_min, y_min), (x_max, y_max),
        COLOR_ORANGE, BOX_THICKNESS
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


def draw_centered_text(frame, text, font=cv2.FONT_HERSHEY_SIMPLEX, scale=3, thickness=4, color=(255, 255, 255)):
    """Draw text perfectly centered on the given frame."""
    frame_height, frame_width = frame.shape[:2]
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    
    # Calculate center position
    x = (frame_width - text_width) // 2
    y = (frame_height + text_height) // 2
    
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
    # State can be 'LIVE' or 'FROZEN'
    state = 'LIVE'
    cycle_start = time.time()
    
    last_known_bbox = None
    locked_bbox = None
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
            # 1. ALWAYS RUN HAND DETECTION for tracking OR for PiP box update
            # (Though in frozen state, the bbox is locked)
            # ---------------------------------------------------------
            all_hands = detect_all_hands(landmarker, frame, timestamp_ms)

            if state == 'LIVE':
                if all_hands is not None:
                    fingertip_points = collect_all_fingertips(
                        all_hands, frame_width, frame_height
                    )
                    bbox = compute_padded_bounding_box(
                        fingertip_points, frame_width, frame_height
                    )
                    if is_valid_bounding_box(bbox):
                        last_known_bbox = bbox
            else:
                # Optional: Handle things if we want live tracking inside PiP?
                # User asked that "where inside the box I can still move around"
                # so the box itself is locked but the user sees their live camera feed inside it.
                pass

            # ---------------------------------------------------------
            # 2. STATE LOGIC & RENDERING
            # ---------------------------------------------------------
            display_frame = frame.copy()

            if state == 'LIVE':
                elapsed = time.time() - cycle_start
                remaining = max(0.0, LIVE_DURATION_SECONDS - elapsed)

                # Draw the grid if tracked
                if last_known_bbox is not None:
                    x_min, y_min, x_max, y_max = last_known_bbox
                    cv2.rectangle(
                        display_frame, (x_min, y_min), (x_max, y_max),
                        COLOR_GREEN, BOX_THICKNESS
                    )
                
                # Center Countdown ONLY in the last 5 seconds
                if remaining <= 5:
                    draw_centered_text(display_frame, f"{int(remaining)+1}")

                if elapsed >= LIVE_DURATION_SECONDS:
                    # Time's up! Transition to FROZEN
                    state = 'FROZEN'
                    frozen_background = frame.copy()
                    
                    if last_known_bbox is not None:
                        locked_bbox = last_known_bbox
                        print(f"[INFO] Freezing! Box locked: {locked_bbox}")
                    else:
                        # Fallback box if no hand was present
                        margin_x = frame_width // 4
                        margin_y = frame_height // 4
                        locked_bbox = (margin_x, margin_y, 
                                       frame_width - margin_x, 
                                       frame_height - margin_y)
                        print("[WARN] No hands. Using fallback freeze box.")

            elif state == 'FROZEN':
                # Background is frozen, inside locked_bbox is live feed
                display_frame = composite_live_roi(
                    frozen_background, frame, locked_bbox
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
                last_known_bbox = None

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
