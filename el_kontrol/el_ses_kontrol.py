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

# MediaPipe için UTF-8 encoding zorla
os.environ['PYTHONIOENCODING'] = 'utf-8'
if hasattr(sys, 'setdefaultencoding'):
    sys.setdefaultencoding('utf-8')

import mediapipe as mp

# === Ayarlar ===
cam_w, cam_h = 1280, 720
is_muted = False
previous_volume = -20.0

# El hareketi takibi için
gesture_history = []
gesture_threshold = 100  # Hareket için minimum piksel
gesture_frames = 5  # Kaç frame içinde hareket olmalı
last_gesture_time = 0

# İki el kontrolü için
two_hands_mode = False
last_distance = 0
min_distance = 100  # Minimum el mesafesi (piksel)
max_distance = 600  # Maximum el mesafesi (piksel)


# === Ses kontrol ===
devices = AudioUtilities.GetSpeakers()
volume = devices.EndpointVolume
min_vol, max_vol, _ = volume.GetVolumeRange()

# === Mikrofon ===
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
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# === Kamera ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)

def draw_gradient_background(img):
    """Gradient arka plan çizer"""
    for i in range(cam_h):
        ratio = i / cam_h
        # Mor-mavi gradient
        b = int(180 * (1 - ratio) + 40 * ratio)
        g = int(120 * (1 - ratio) + 40 * ratio)
        r = int(150 * (1 - ratio) + 60 * ratio)
        img[i, :] = [b, g, r]
    return img


def handle_gesture(gesture_type):
    """El hareketlerine göre komut çalıştırır"""
    global last_gesture_time, previous_volume, is_muted
    now = time.time()
    if now - last_gesture_time < 1.5:
        return
    
    if gesture_type == 'swipe_right':
        media_next()
        print("👉 Sağa kaydırma: Sonraki şarkı")
    elif gesture_type == 'swipe_left':
        media_prev()
        print("👈 Sola kaydırma: Önceki şarkı")
    elif gesture_type == 'swipe_up':
        volume.SetMasterVolumeLevel(min(volume.GetMasterVolumeLevel() + 3.0, max_vol), None)
        print("👆 Yukarı kaydırma: Ses artırıldı")
    elif gesture_type == 'swipe_down':
        volume.SetMasterVolumeLevel(max(volume.GetMasterVolumeLevel() - 3.0, min_vol), None)
        print("👇 Aşağı kaydırma: Ses azaltıldı")
    elif gesture_type == 'fist':
        curr_vol = volume.GetMasterVolumeLevel()
        if curr_vol <= min_vol + 1.0:
            volume.SetMasterVolumeLevel(previous_volume, None)
            is_muted = False
            print("✊ Yumruk: Sessizden çıktı")
        else:
            previous_volume = curr_vol
            volume.SetMasterVolumeLevel(min_vol, None)
            is_muted = True
            print("✊ Yumruk: Sessize alındı")
    
    last_gesture_time = now

def is_palm_open(fingers):
    return sum(fingers) >= 4

def is_fist(fingers):
    return sum(fingers) == 0

def detect_swipe_gesture(history):
    """El kaydırma hareketlerini algılar"""
    if len(history) < gesture_frames:
        return None
    
    recent = history[-gesture_frames:]
    start_x, start_y = recent[0]
    end_x, end_y = recent[-1]
    
    dx = end_x - start_x
    dy = end_y - start_y
    
    # Yatay hareket (sağ/sol)
    if abs(dx) > gesture_threshold and abs(dx) > abs(dy) * 2:
        return 'swipe_right' if dx > 0 else 'swipe_left'
    
    # Dikey hareket (yukarı/aşağı)
    if abs(dy) > gesture_threshold and abs(dy) > abs(dx) * 2:
        return 'swipe_up' if dy < 0 else 'swipe_down'
    
    return None

def get_hand_center(landmarks, w, h):
    """Elin merkez noktasını hesaplar"""
    x_coords = [lm.x * w for lm in landmarks]
    y_coords = [lm.y * h for lm in landmarks]
    center_x = int(sum(x_coords) / len(x_coords))
    center_y = int(sum(y_coords) / len(y_coords))
    return (center_x, center_y)

def control_volume_by_distance(distance):
    """İki el arası mesafeye göre ses seviyesini ayarlar"""
    # Mesafeyi ses seviyesine map et
    distance = max(min_distance, min(distance, max_distance))
    volume_ratio = (distance - min_distance) / (max_distance - min_distance)
    
    # dB cinsinden ses seviyesi hesapla
    target_volume = min_vol + (max_vol - min_vol) * volume_ratio
    volume.SetMasterVolumeLevel(target_volume, None)
    
    return int(volume_ratio * 100)

action_delay = 0.0

while True:
    ret, img = cap.read()
    if not ret:
        continue
    img = cv2.flip(img, 1)
    
    # Modern gradient arka plan
    img_display = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
    img_display = draw_gradient_background(img_display)
    
    # Kamera görüntüsünü RENKLİ ve saydam olarak ekle
    img_small = cv2.resize(img, (cam_w, cam_h))
    img_display = cv2.addWeighted(img_display, 0.6, img_small, 0.4, 0)
    
    # İşleme için siyah-beyaz kullan
    img_gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    img_small_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    
    results = hands.process(img_small_rgb)
    
    # Ses durumunu kontrol et
    curr_volume_level = volume.GetMasterVolumeLevel()
    is_muted = curr_volume_level <= min_vol + 1.0
    
    # El tespiti
    current_button = None
    num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
    
    # İKİ EL MODU - Mesafe ile ses kontrolü
    if num_hands == 2:
        two_hands_mode = True
        hand1_lms = results.multi_hand_landmarks[0]
        hand2_lms = results.multi_hand_landmarks[1]
        
        # Her iki eli de çiz
        for idx, handLms in enumerate(results.multi_hand_landmarks):
            color = (0, 255, 0) if idx == 0 else (0, 0, 255)  # Sol yeşil, sağ kırmızı
            mp_draw.draw_landmarks(
                img_display, 
                handLms, 
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(255, 255, 0), thickness=3)
            )
        
        # Ellerin merkez noktalarını bul
        center1 = get_hand_center(hand1_lms.landmark, cam_w, cam_h)
        center2 = get_hand_center(hand2_lms.landmark, cam_w, cam_h)
        
        # Mesafeyi hesapla
        distance = int(np.sqrt((center2[0] - center1[0])**2 + (center2[1] - center1[1])**2))
        
        # Çizgi çiz
        cv2.line(img_display, center1, center2, (255, 0, 255), 3)
        cv2.circle(img_display, center1, 10, (0, 255, 0), -1)
        cv2.circle(img_display, center2, 10, (0, 0, 255), -1)
        
        # Ses seviyesini ayarla
        volume_percent = control_volume_by_distance(distance)
        
        # Bilgi göster
        mid_x = (center1[0] + center2[0]) // 2
        mid_y = (center1[1] + center2[1]) // 2
        cv2.putText(img_display, f"Mesafe: {distance}px", (mid_x - 80, mid_y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img_display, f"Ses: %{volume_percent}", (mid_x - 60, mid_y + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Büyük uyarı
        cv2.putText(img_display, "IKI EL MODU - SES KONTROLU", (cam_w//2 - 220, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    
    # TEK EL MODU - Normal kontroller
    elif num_hands == 1:
        two_hands_mode = False
        handLms = results.multi_hand_landmarks[0]
        
        # El iskeletini çiz (tüm parmaklar)
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
        
        # İşaret parmağı pozisyonunu kaydet (hareket takibi için)
        index_finger_pos = (lm_list[8][1], lm_list[8][2])
        gesture_history.append(index_finger_pos)
        if len(gesture_history) > 15:
            gesture_history.pop(0)
        
        # Yumruk kontrolü
        if is_fist(fingers):
            handle_gesture('fist')
            cv2.putText(img_display, "YUMRUK - SESSIZ/AC", (50, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Açık el - hareket kontrolü
        elif is_palm_open(fingers):
            x, y = index_finger_pos
            
            # El pozisyon göstergesi
            cv2.circle(img_display, (x, y), 15, (0, 255, 255), -1)
            cv2.circle(img_display, (x, y), 18, (255, 255, 255), 2)
            
            # Hareket algılama
            gesture = detect_swipe_gesture(gesture_history)
            if gesture:
                handle_gesture(gesture)
                gesture_history.clear()
        else:
            gesture_history.clear()
    else:
        # El yok
        two_hands_mode = False
        gesture_history.clear()
    
    # Başlık ve yönergeler
    cv2.putText(img_display, "EL HAREKET KONTROL", (cam_w//2 - 220, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    # Hareket yönergeleri
    if two_hands_mode:
        instructions = [
            "IKI EL: Yaklastir/Uzaklastir = Ses Seviyesi"
        ]
    else:
        instructions = [
            "TEK EL: Sag/Sol = Sarki Degistir | Yukari/Asagi = Ses | Yumruk = Sessiz",
            "IKI EL: Elleri ac ve yaklastir/uzaklastir = Hassas Ses Kontrolu"
        ]
    
    for i, text in enumerate(instructions):
        cv2.putText(img_display, text, (15, cam_h - 60 + i*30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 255, 200), 2)
    
    cv2.imshow("🎛️ El Kontrollü Medya Paneli", img_display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()