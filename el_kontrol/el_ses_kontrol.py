import cv2
import numpy as np
import pyautogui
import time
import speech_recognition as sr
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import ctypes
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import os
import sys
from dotenv import load_dotenv
load_dotenv()

os.environ['PYTHONIOENCODING'] = 'utf-8'
if hasattr(sys, 'setdefaultencoding'):
    sys.setdefaultencoding('utf-8')

import mediapipe as mp

cam_w, cam_h = 1280, 720
is_muted = False
previous_volume = -20.0

icon_path = os.path.join(os.path.dirname(__file__), "icons")
icon_size = (70, 70)

center_x, center_y = cam_w // 2, cam_h // 2

button_y = 150
button_spacing = 150

button_positions = {
    'prev': (center_x - button_spacing * 1.5, button_y),
    'playpause': (center_x - button_spacing * 0.5, button_y),
    'next': (center_x + button_spacing * 0.5, button_y),
    'spotifysearch': (center_x + button_spacing * 1.5, button_y)
}

button_radius = 50

last_finger_action_time = 0
finger_action_cooldown = 1.0

two_hands_mode = False
last_distance = 0
min_distance = 100
max_distance = 600
last_two_hands_action_time = 0
two_hands_action_cooldown = 2.0

last_volume_set_time = 0
volume_update_interval = 0.1
min_distance_change = 30
last_stable_distance = 0

two_hands_counter = 0
one_hand_counter = 0
hands_detection_threshold = 3

devices = AudioUtilities.GetSpeakers()
volume = devices.EndpointVolume
min_vol, max_vol, _ = volume.GetVolumeRange()

mic = None
previous_mic_volume = -15.0
try:
    for dev in AudioUtilities.GetAllDevices():
        try:
            is_input = dev.properties.get("{a45c254e-df1c-4efd-8020-67d146a850e0},2") == 1
            is_active = dev.state == 1
            if is_input and is_active:
                mic = dev.EndpointVolume
                break
        except:
            continue
except:
    pass

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
    redirect_uri="http://127.0.0.1:8888/callback",
    scope="user-modify-playback-state user-read-playback-state"
))

def media_next():
    ctypes.windll.user32.keybd_event(0xB0, 0, 0, 0)
    ctypes.windll.user32.keybd_event(0xB0, 0, 2, 0)

def media_prev():
    ctypes.windll.user32.keybd_event(0xB1, 0, 0, 0)
    ctypes.windll.user32.keybd_event(0xB1, 0, 2, 0)

def media_play_pause():
    ctypes.windll.user32.keybd_event(0xB3, 0, 0, 0)
    ctypes.windll.user32.keybd_event(0xB3, 0, 2, 0)

def listen_and_play_spotify():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Dinleniyor... Şarkı adını söyle (5 saniye):")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
        except sr.WaitTimeoutError:
            print("⏱️ Zaman aşımı, ses algılanmadı.")
            return

    try:
        song = recognizer.recognize_google(audio, language="tr-TR")
        print(f"🎶 Algılanan şarkı: {song}")
        results = sp.search(q=song, limit=1, type='track')
        if results['tracks']['items']:
            uri = results['tracks']['items'][0]['uri']
            sp.start_playback(uris=[uri])
            print(f"🎧 Spotify'da çalınıyor: {song}")
        else:
            print("❌ Şarkı bulunamadı.")
    except sr.UnknownValueError:
        print("🧏 Ses anlaşılamadı.")
    except sr.RequestError as e:
        print(f"🌐 API Hatası: {e}")
    except Exception as e:
        print(f"⚠️ Beklenmeyen hata: {e}")


def toggle_microphone():
    global previous_mic_volume
    if not mic:
        print("Mikrofon bulunamadı")
        return
    curr = mic.GetMasterVolumeLevel()
    mic_min, mic_max, _ = mic.GetVolumeRange()
    if curr <= mic_min + 1.0:
        mic.SetMasterVolumeLevel(previous_mic_volume, None)
    else:
        previous_mic_volume = curr
        mic.SetMasterVolumeLevel(mic_min, None)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)

icons = {}
for name in ["next", "prev", "playpause", "spotifysearch"]:
    path = os.path.join(icon_path, name + ".png")
    icon = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if icon is not None:
        icons[name] = cv2.resize(icon, icon_size)

def overlay_icon(bg, icon, pos, scale=1.0):
    if scale != 1.0:
        new_size = (int(icon_size[0] * scale), int(icon_size[1] * scale))
        icon = cv2.resize(icon, new_size)
    
    h, w = icon.shape[:2]
    x, y = int(pos[0] - w // 2), int(pos[1] - h // 2)
    
    if icon.shape[2] == 4:
        for i in range(h):
            for j in range(w):
                if 0 <= y+i < bg.shape[0] and 0 <= x+j < bg.shape[1]:
                    if icon[i, j][3] > 0:
                        alpha = icon[i, j][3] / 255.0
                        bg[y+i, x+j] = (1 - alpha) * bg[y+i, x+j] + alpha * icon[i, j][:3]
    else:
        if y >= 0 and x >= 0 and y+h <= bg.shape[0] and x+w <= bg.shape[1]:
            bg[y:y+h, x:x+w] = icon

def get_button_at_position(x, y):
    for key, pos in button_positions.items():
        dist = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
        if dist < button_radius:
            return key
    return None

def draw_gradient_background(img):
    for i in range(cam_h):
        ratio = i / cam_h
        b = int(180 * (1 - ratio) + 40 * ratio)
        g = int(120 * (1 - ratio) + 40 * ratio)
        r = int(150 * (1 - ratio) + 60 * ratio)
        img[i, :] = [b, g, r]
    return img


def handle_gesture(gesture_type):
    global last_gesture_time, previous_volume, is_muted
    now = time.time()
    if now - last_gesture_time < gesture_cooldown:
        return
    
    if gesture_type == 'swipe_right':
        media_next()
        print("👉 Sağa kaydırma: Sonraki şarkı")
    elif gesture_type == 'swipe_left':
        media_prev()
        print("👈 Sola kaydırma: Önceki şarkı")
    elif gesture_type == 'swipe_up':
        media_play_pause()
        print("👆 Yukarı kaydırma: Oynat/Duraklat")
    elif gesture_type == 'swipe_down':
        listen_and_play_spotify()
        print("👇 Aşağı kaydırma: Spotify arama")
    
    last_gesture_time = now

def handle_button_action(button_key):
    global last_finger_action_time
    now = time.time()
    
    if now - last_finger_action_time < finger_action_cooldown:
        return None
    
    action = None
    if button_key == 'prev':
        media_prev()
        action = "⏮️ Önceki şarkı"
        last_finger_action_time = now
    elif button_key == 'playpause':
        media_play_pause()
        action = "⏯️ Oynat/Duraklat"
        last_finger_action_time = now
    elif button_key == 'next':
        media_next()
        action = "⏭️ Sonraki şarkı"
        last_finger_action_time = now
    elif button_key == 'spotifysearch':
        listen_and_play_spotify()
        action = "🎵 Spotify Sesli Arama"
        last_finger_action_time = now
    
    if action:
        print(action)
    return action

def handle_two_hands_gesture(hand1_fingers, hand2_fingers):
    return None

def is_palm_open(fingers):
    return sum(fingers) >= 4

def get_finger_count(fingers):
    return sum(fingers)

def is_fist(fingers):
    return sum(fingers) == 0

def get_hand_center(landmarks, w, h):
    x_coords = [lm.x * w for lm in landmarks]
    y_coords = [lm.y * h for lm in landmarks]
    center_x = int(sum(x_coords) / len(x_coords))
    center_y = int(sum(y_coords) / len(y_coords))
    return (center_x, center_y)

def control_volume_by_distance(distance):
    global last_volume_set_time, last_stable_distance
    
    now = time.time()
    
    if now - last_volume_set_time < volume_update_interval:
        distance = max(min_distance, min(distance, max_distance))
        volume_ratio = (distance - min_distance) / (max_distance - min_distance)
        return int(volume_ratio * 100)
    
    if last_stable_distance > 0:
        distance_diff = abs(distance - last_stable_distance)
        if distance_diff < min_distance_change:
            distance = max(min_distance, min(distance, max_distance))
            volume_ratio = (distance - min_distance) / (max_distance - min_distance)
            return int(volume_ratio * 100)
    
    distance = max(min_distance, min(distance, max_distance))
    volume_ratio = (distance - min_distance) / (max_distance - min_distance)
    
    target_volume = min_vol + (max_vol - min_vol) * volume_ratio
    volume.SetMasterVolumeLevel(target_volume, None)
    
    last_volume_set_time = now
    last_stable_distance = distance
    
    return int(volume_ratio * 100)

action_delay = 0.0

while True:
    ret, img = cap.read()
    if not ret:
        continue
    img = cv2.flip(img, 1)
    
    img_display = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
    img_display = draw_gradient_background(img_display)
    
    img_small = cv2.resize(img, (cam_w, cam_h))
    img_display = cv2.addWeighted(img_display, 0.6, img_small, 0.4, 0)
    
    img_gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    img_small_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    
    results = hands.process(img_small_rgb)
    
    curr_volume_level = volume.GetMasterVolumeLevel()
    is_muted = curr_volume_level <= min_vol + 1.0
    
    current_button = None
    num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
    
    if num_hands == 2:
        two_hands_counter += 1
        one_hand_counter = 0
    elif num_hands == 1:
        one_hand_counter += 1
        two_hands_counter = 0
    else:
        two_hands_counter = 0
        one_hand_counter = 0
    
    if two_hands_counter >= hands_detection_threshold:
        two_hands_mode = True
    elif one_hand_counter >= hands_detection_threshold or num_hands == 0:
        two_hands_mode = False
    
    if two_hands_mode and num_hands == 2:
        two_hands_mode = True
        hand1_lms = results.multi_hand_landmarks[0]
        hand2_lms = results.multi_hand_landmarks[1]
        
        for idx, handLms in enumerate(results.multi_hand_landmarks):
            color = (0, 255, 0) if idx == 0 else (0, 0, 255)
            mp_draw.draw_landmarks(
                img_display, 
                handLms, 
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(255, 255, 0), thickness=3)
            )
        
        center1 = get_hand_center(hand1_lms.landmark, cam_w, cam_h)
        center2 = get_hand_center(hand2_lms.landmark, cam_w, cam_h)
        
        distance = int(np.sqrt((center2[0] - center1[0])**2 + (center2[1] - center1[1])**2))
        
        cv2.line(img_display, center1, center2, (255, 0, 255), 3)
        cv2.circle(img_display, center1, 10, (0, 255, 0), -1)
        cv2.circle(img_display, center2, 10, (0, 0, 255), -1)
        
        volume_percent = control_volume_by_distance(distance)
        
        mid_x = (center1[0] + center2[0]) // 2
        mid_y = (center1[1] + center2[1]) // 2
        cv2.putText(img_display, f"Mesafe: {distance}px", (mid_x - 80, mid_y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img_display, f"Ses: %{volume_percent}", (mid_x - 60, mid_y + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(img_display, "IKI EL MODU - SES KONTROLU", (cam_w//2 - 220, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    
    elif not two_hands_mode and num_hands == 1:
        two_hands_mode = False
        handLms = results.multi_hand_landmarks[0]
        
        mp_draw.draw_landmarks(
            img_display, 
            handLms, 
            mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
            mp_draw.DrawingSpec(color=(255, 255, 0), thickness=3)
        )
        
        lm_list = [(id, int(lm.x * cam_w), int(lm.y * cam_h)) for id, lm in enumerate(handLms.landmark)]
        finger_tips = [4, 8, 12, 16, 20]
        fingers = [1 if lm_list[4][1] > lm_list[3][1] else 0]
        for tip in finger_tips[1:]:
            fingers.append(1 if lm_list[tip][2] < lm_list[tip - 2][2] else 0)
        
        finger_count = get_finger_count(fingers)
        
        index_finger_pos = (lm_list[8][1], lm_list[8][2])
        
        cv2.circle(img_display, index_finger_pos, 15, (0, 255, 255), -1)
        cv2.circle(img_display, index_finger_pos, 18, (255, 255, 255), 2)
        
        current_button = None
        if finger_count >= 3:
            current_button = get_button_at_position(index_finger_pos[0], index_finger_pos[1])
            if current_button:
                button_action = handle_button_action(current_button)
                if button_action:
                    cv2.putText(img_display, button_action, (50, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 100, 255), 2)
        
        cv2.putText(img_display, f"Parmak: {finger_count}", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    else:
        current_button = None
        if num_hands == 0:
            two_hands_counter = 0
            one_hand_counter = 0
    
    if not two_hands_mode:
        for key in ['prev', 'playpause', 'next', 'spotifysearch']:
            if key not in icons:
                continue
            
            pos = button_positions[key]
            is_active = (current_button == key) if 'current_button' in locals() else False
            
            button_color = (100, 255, 100) if is_active else (80, 120, 180)
            
            cv2.circle(img_display, (int(pos[0]), int(pos[1])), button_radius, button_color, -1)
            cv2.circle(img_display, (int(pos[0]), int(pos[1])), button_radius, (255, 255, 255), 3)
            
            scale = 1.2 if is_active else 1.0
            overlay_icon(img_display, icons[key], pos, scale)
    
    cv2.putText(img_display, "EL HAREKET KONTROL", (cam_w//2 - 220, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    if two_hands_mode:
        instructions = [
            "IKI EL MODU: Elleri yaklastir/uzaklastir = SES KONTROLU"
        ]
    else:
        instructions = [
            "MEDYA: Acik elle (3+ parmak) butonlara dokun",
            "SES: Iki elinizi goster ve yaklastir/uzaklastir"
        ]
    
    for i, text in enumerate(instructions):
        y_pos = cam_h - 200 + i*30
        cv2.putText(img_display, text, (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
    
    cv2.imshow("🎛️ El Kontrollü Medya Paneli", img_display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()