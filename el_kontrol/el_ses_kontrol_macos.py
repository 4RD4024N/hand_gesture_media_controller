import cv2
import numpy as np
import pyautogui
import time
import speech_recognition as sr
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import subprocess
import os
from dotenv import load_dotenv

# MediaPipe Tasks API
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

load_dotenv()

# === Settings ===
cam_w, cam_h = 640, 480  # Smaller size to fit on screen
cols, rows = 3, 3  # 3x3 grid for better usability
cell_w = cam_w // cols
cell_h = cam_h // rows
icon_size = (80, 80)  # Larger icons for visibility
is_muted = False
previous_volume = 50

icon_path = os.path.join(os.path.dirname(__file__), "icons")
model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

# === Gesture confirmation ===
hover_start_time = {}
HOVER_CONFIRM_TIME = 0.8  # Seconds to hold position before triggering
last_action_time = 0
ACTION_COOLDOWN = 1.5  # Seconds between actions

# === macOS Volume Control ===
def get_volume():
    """Get current volume level (0-100)"""
    try:
        result = subprocess.run(
            ["osascript", "-e", "output volume of (get volume settings)"],
            capture_output=True, text=True, timeout=1
        )
        return int(result.stdout.strip())
    except:
        return 50

def set_volume(level):
    """Set volume level (0-100)"""
    level = max(0, min(100, level))
    subprocess.run(["osascript", "-e", f"set volume output volume {level}"], timeout=1)

def is_volume_muted():
    """Check if volume is muted"""
    try:
        result = subprocess.run(
            ["osascript", "-e", "output muted of (get volume settings)"],
            capture_output=True, text=True, timeout=1
        )
        return result.stdout.strip() == "true"
    except:
        return False

def toggle_mute():
    """Toggle mute state"""
    try:
        muted = is_volume_muted()
        subprocess.run(["osascript", "-e", f"set volume output muted {str(not muted).lower()}"], timeout=1)
    except:
        pass

# === Spotify Auth ===
sp = None
try:
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if client_id and client_secret:
        sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri="http://127.0.0.1:8888/callback",
            scope="user-modify-playback-state user-read-playback-state"
        ))
        print("✅ Spotify configured")
except Exception as e:
    print(f"⚠️ Spotify not configured: {e}")

# === Media Controls (using osascript for macOS - more reliable) ===
def media_next():
    try:
        # Try Spotify first, then Music
        subprocess.run(["osascript", "-e", 'tell application "Spotify" to next track'], 
                      timeout=2, capture_output=True)
    except:
        pass
    time.sleep(0.1)

def media_prev():
    try:
        # Try Spotify first, then Music
        subprocess.run(["osascript", "-e", 'tell application "Spotify" to previous track'], 
                      timeout=2, capture_output=True)
    except:
        pass
    time.sleep(0.1)

def media_play_pause():
    try:
        # Try Spotify play/pause
        subprocess.run(["osascript", "-e", 'tell application "Spotify" to playpause'], 
                      timeout=2, capture_output=True)
    except:
        # Fallback: press space key
        try:
            pyautogui.press('space')
        except:
            pass
    time.sleep(0.1)

def listen_and_play_spotify():
    if sp is None:
        print("⚠️ Spotify not configured")
        return
    
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening... Say the song name (5 seconds):")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
        except sr.WaitTimeoutError:
            print("⏱️ Timeout")
            return

    try:
        song = recognizer.recognize_google(audio, language="tr-TR")
        print(f"🎶 Detected song: {song}")
        results = sp.search(q=song, limit=1, type='track')
        if results['tracks']['items']:
            uri = results['tracks']['items'][0]['uri']
            sp.start_playback(uris=[uri])
            print(f"🎧 Playing: {song}")
        else:
            print("❌ Song not found.")
    except Exception as e:
        print(f"⚠️ Error: {e}")

# === MediaPipe Setup ===
if not os.path.exists(model_path):
    print("❌ Error: hand_landmarker.task model not found!")
    exit(1)

latest_result = None

def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
    result_callback=result_callback
)

landmarker = vision.HandLandmarker.create_from_options(options)

# === Camera ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)

# === Load Icons ===
icons = {}
for name in ["next", "prev", "playpause", "spotifysearch", "volumedown", "volumeoff", "volumeon", "volumeup"]:
    path = os.path.join(icon_path, name + ".png")
    icon = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if icon is not None:
        icons[name] = cv2.resize(icon, icon_size)
        print(f"✅ Loaded icon: {name}")
    else:
        print(f"⚠️ Icon not found: {path}")

def overlay_icon(bg, icon, pos):
    x, y = pos
    h, w = icon.shape[:2]
    if x < 0 or y < 0 or x + w > bg.shape[1] or y + h > bg.shape[0]:
        return
    if icon.shape[2] == 4:
        alpha = icon[:, :, 3] / 255.0
        for c in range(3):
            bg[y:y+h, x:x+w, c] = (alpha * icon[:, :, c] + (1 - alpha) * bg[y:y+h, x:x+w, c]).astype(np.uint8)
    else:
        bg[y:y+h, x:x+w] = icon[:, :, :3]

# Action grid layout (row, col) -> (action_name, icon_name, label)
GRID_ACTIONS = {
    (0, 0): ("vol_down", "volumedown", "Vol -"),
    (0, 1): ("play_pause", "playpause", "Play/Pause"),
    (0, 2): ("vol_up", "volumeup", "Vol +"),
    (1, 0): ("prev", "prev", "Previous"),
    (1, 1): ("spotify", "spotifysearch", "Spotify"),
    (1, 2): ("next", "next", "Next"),
    (2, 0): ("prev2", "prev", "2x Back"),  # Go back 2 songs
    (2, 1): ("mute", "volumeon", "Mute"),
    (2, 2): None,  # Empty cell
}

def execute_action(action_name):
    global last_action_time
    now = time.time()
    if now - last_action_time < ACTION_COOLDOWN:
        return False
    
    last_action_time = now
    
    if action_name == "vol_down":
        current = get_volume()
        set_volume(current - 10)
        print(f"🔉 Volume: {max(0, current - 10)}%")
    elif action_name == "vol_up":
        current = get_volume()
        set_volume(current + 10)
        print(f"🔊 Volume: {min(100, current + 10)}%")
    elif action_name == "play_pause":
        media_play_pause()
        print("⏯️ Play/Pause")
    elif action_name == "prev":
        media_prev()
        print("⏮️ Previous Track")
    elif action_name == "prev2":
        media_prev()
        time.sleep(0.3)
        media_prev()
        print("⏮️⏮️ Back 2 Tracks")
    elif action_name == "next":
        media_next()
        print("⏭️ Next Track")
    elif action_name == "spotify":
        listen_and_play_spotify()
    elif action_name == "mute":
        toggle_mute()
        if is_volume_muted():
            print("🔇 Muted")
        else:
            print("🔊 Unmuted")
    return True

def is_palm_open(fingers):
    return sum(fingers) >= 4

frame_timestamp = 0
mute_check_timer = 0

print("\n🎛️ Hand Gesture Media Controller (macOS)")
print("=" * 40)
print("Hold your OPEN PALM over a button for 0.8s to activate")
print("Press 'q' to quit")
print("=" * 40 + "\n")

while True:
    ret, img = cap.read()
    if not ret:
        continue
    
    img = cv2.flip(img, 1)
    
    # Create display image with grayscale background
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Process hand detection
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    frame_timestamp += 33
    landmarker.detect_async(mp_image, frame_timestamp)
    
    # Check mute state less frequently (every 30 frames)
    mute_check_timer += 1
    if mute_check_timer >= 30:
        is_muted = is_volume_muted()
        mute_check_timer = 0
    
    # Draw grid lines
    for i in range(1, cols):
        cv2.line(img_display, (i * cell_w, 0), (i * cell_w, cam_h), (100, 100, 100), 3)
    for j in range(1, rows):
        cv2.line(img_display, (0, j * cell_h), (cam_w, j * cell_h), (100, 100, 100), 3)
    
    current_cell = None
    hover_progress = 0
    
    # Process hand landmarks
    if latest_result and latest_result.hand_landmarks:
        hand_landmarks = latest_result.hand_landmarks[0]
        lm_list = [(i, int(lm.x * cam_w), int(lm.y * cam_h)) for i, lm in enumerate(hand_landmarks)]
        
        # Draw hand landmarks
        for idx, x, y in lm_list:
            cv2.circle(img_display, (x, y), 5, (0, 255, 255), -1)
        
        # Finger detection
        finger_tips = [4, 8, 12, 16, 20]
        fingers = [1 if lm_list[4][1] > lm_list[3][1] else 0]
        for tip in finger_tips[1:]:
            fingers.append(1 if lm_list[tip][2] < lm_list[tip - 2][2] else 0)
        
        if is_palm_open(fingers):
            # Use center of palm (landmark 9) for more stable tracking
            palm_x, palm_y = lm_list[9][1], lm_list[9][2]
            col = int(palm_x // cell_w)
            row = int(palm_y // cell_h)
            
            if 0 <= col < cols and 0 <= row < rows:
                current_cell = (row, col)
                action_data = GRID_ACTIONS.get(current_cell)
                
                if action_data:
                    # Track hover time
                    if current_cell not in hover_start_time:
                        hover_start_time[current_cell] = time.time()
                    
                    hover_duration = time.time() - hover_start_time[current_cell]
                    hover_progress = min(1.0, hover_duration / HOVER_CONFIRM_TIME)
                    
                    # Draw progress indicator
                    cell_x = col * cell_w
                    cell_y = row * cell_h
                    
                    if hover_progress < 1.0:
                        # Yellow while hovering
                        overlay = img_display.copy()
                        cv2.rectangle(overlay, (cell_x, cell_y), (cell_x + cell_w, cell_y + cell_h), (0, 255, 255), -1)
                        cv2.addWeighted(overlay, 0.3, img_display, 0.7, 0, img_display)
                        
                        # Progress bar
                        bar_width = int(cell_w * hover_progress)
                        cv2.rectangle(img_display, (cell_x, cell_y + cell_h - 10), 
                                     (cell_x + bar_width, cell_y + cell_h), (0, 200, 0), -1)
                    
                    # Trigger action when hover confirmed
                    if hover_progress >= 1.0:
                        overlay = img_display.copy()
                        cv2.rectangle(overlay, (cell_x, cell_y), (cell_x + cell_w, cell_y + cell_h), (0, 255, 0), -1)
                        cv2.addWeighted(overlay, 0.4, img_display, 0.6, 0, img_display)
                        
                        if execute_action(action_data[0]):
                            hover_start_time.clear()
                    
                    # Draw palm position indicator
                    cv2.circle(img_display, (palm_x, palm_y), 20, (255, 0, 255), -1)
    
    # Clear hover times for cells not currently hovered
    if current_cell:
        for cell in list(hover_start_time.keys()):
            if cell != current_cell:
                del hover_start_time[cell]
    else:
        hover_start_time.clear()
    
    # Draw icons and labels
    for (row, col), action_data in GRID_ACTIONS.items():
        if action_data is None:
            continue
        
        action_name, icon_name, label = action_data
        
        # Update mute icon based on state
        if action_name == "mute":
            icon_name = "volumeoff" if is_muted else "volumeon"
        
        if icon_name in icons:
            icon = icons[icon_name]
            x = col * cell_w + (cell_w - icon_size[0]) // 2
            y = row * cell_h + (cell_h - icon_size[1]) // 2 - 15
            overlay_icon(img_display, icon, (x, y))
        
        # Draw label
        label_x = col * cell_w + cell_w // 2
        label_y = row * cell_h + cell_h // 2 + icon_size[1] // 2 + 10
        cv2.putText(img_display, label, (label_x - len(label) * 5, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw status bar
    cv2.rectangle(img_display, (0, cam_h - 40), (cam_w, cam_h), (50, 50, 50), -1)
    status = f"Volume: {get_volume() if mute_check_timer == 0 else '--'}%  |  "
    status += "Muted" if is_muted else "Sound On"
    status += "  |  Hold palm 0.8s to activate"
    cv2.putText(img_display, status, (10, cam_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    cv2.imshow("Hand Gesture Media Controller", img_display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

landmarker.close()
cap.release()
cv2.destroyAllWindows()
