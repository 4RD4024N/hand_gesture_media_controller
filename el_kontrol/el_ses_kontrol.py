import cv2
import mediapipe as mp
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
from dotenv import load_dotenv
load_dotenv()

# === Ayarlar ===
cam_w, cam_h = 1280, 720
cols, rows = 3, 4
cell_w = cam_w // cols
cell_h = cam_h // rows
icon_size = (64, 64)
is_muted = False

icon_path = os.path.join(os.path.dirname(__file__), "icons")


# === Ses kontrol ===
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
min_vol, max_vol, _ = volume.GetVolumeRange()

# === Mikrofon ===
mic = None
previous_mic_volume = -15.0
for dev in AudioUtilities.GetAllDevices():
    is_input = dev.properties.get("{a45c254e-df1c-4efd-8020-67d146a850e0},2") == 1
    is_active = dev.state == 1
    if is_input and is_active:
        mic_interface = dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        mic = cast(mic_interface, POINTER(IAudioEndpointVolume))
        break

# === Spotify Auth ===
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
    redirect_uri="http://127.0.0.1:8888/callback",
    scope="user-modify-playback-state user-read-playback-state"
))

# === Sistem medya tuşları ===
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

# === MediaPipe ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# === Kamera ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)

# === Ikonları yükle ===
icons = {}
for name in ["next", "prev", "playpause", "spotifysearch","volumedown","volumeoff","volumeon","volumeup"]:
    path = os.path.join(icon_path, name + ".png")
    icon = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if icon is not None:
        icons[name] = cv2.resize(icon, icon_size)

def overlay_icon(bg, icon, pos):
    x, y = pos
    h, w = icon.shape[:2]
    if icon.shape[2] == 4:
        for i in range(h):
            for j in range(w):
                if icon[i, j][3] != 0 and y+i < bg.shape[0] and x+j < bg.shape[1]:
                    bg[y+i, x+j] = icon[i, j][:3]
    else:
        bg[y:y+h, x:x+w] = icon


def handle_action(col, row):
    global action_delay, previous_volume, is_muted
    now = time.time()
    if now - action_delay < 1:
        return

    if row == 0:
        if col == 0:
            volume.SetMasterVolumeLevel(max(volume.GetMasterVolumeLevel() - 2.0, min_vol), None)
        elif col == 1:
            media_play_pause()
        elif col == 2:
            volume.SetMasterVolumeLevel(min(volume.GetMasterVolumeLevel() + 2.0, max_vol), None)

    elif row == 1:
        if col == 0:
            media_prev()
        elif col == 1:
            listen_and_play_spotify()
        elif col == 2:
            media_next()

    elif row == 2 and col == 1:
        curr_vol = volume.GetMasterVolumeLevel()
        if curr_vol <= min_vol + 1.0:
            volume.SetMasterVolumeLevel(previous_volume, None)
            is_muted = False
            print(f"🔊 SESSIZDEN CIKTI: {previous_volume:.1f} dB")
        else:
            previous_volume = curr_vol
            volume.SetMasterVolumeLevel(min_vol, None)
            is_muted = True
            print("🔇 SESSIZE ALINDI")

    action_delay = now


        

    action_delay = now

def is_palm_open(fingers):
    return sum(fingers) >= 4

action_delay = time.time()

while True:
    ret, img = cap.read()
    if not ret:
        continue
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   

    img_display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_small = cv2.resize(img, (cam_w, cam_h))
    img_small_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    results = hands.process(img_small_rgb)

    for i in range(1, cols):
        cv2.line(img_display, (i * cell_w, 0), (i * cell_w, cam_h), (150, 150, 150), 2)
    for j in range(1, rows):
        cv2.line(img_display, (0, j * cell_h), (cam_w, j * cell_h), (150, 150, 150), 2)
    

    curr_volume_level = volume.GetMasterVolumeLevel()
    is_muted = curr_volume_level <= min_vol + 1.0

    icon_map = {
      (0, 0): "volumedown", 
     (0, 1): "playpause", 
     (0, 2): "volumeup",
     (1, 0): "prev", 
     (1, 1): "spotifysearch", 
     (1, 2): "next",
     (2, 1): "volumeoff" if is_muted else "volumeon"
        }


    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        lm_list = [(id, int(lm.x * cam_w), int(lm.y * cam_h)) for id, lm in enumerate(handLms.landmark)]
        finger_tips = [4, 8, 12, 16, 20]
        fingers = [1 if lm_list[4][1] > lm_list[3][1] else 0]
        for tip in finger_tips[1:]:
            fingers.append(1 if lm_list[tip][2] < lm_list[tip - 2][2] else 0)

        if is_palm_open(fingers):
            x, y = lm_list[8][1], lm_list[8][2]
            col = int(x // cell_w)
            row = int(y // cell_h)
            cv2.rectangle(img_display, (col * cell_w, row * cell_h), ((col + 1) * cell_w, (row + 1) * cell_h), (0, 255, 0), -1)
            handle_action(col, row)

    for (r, row) in enumerate(range(rows)):
        for (c, col) in enumerate(range(cols)):
            key = icon_map.get((row, col))
            if key in icons:
                icon = icons[key]
                x = col * cell_w + (cell_w - icon_size[0]) // 2
                y = row * cell_h + (cell_h - icon_size[1]) // 2
                overlay_icon(img_display, icon, (x, y))

    cv2.imshow("🎛️ El Kontrollü Medya Paneli", img_display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()