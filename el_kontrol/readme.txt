# ✋🎧 Hand-Gesture Controlled Media Panel

This project is a gesture-based media control panel using your webcam and hand tracking. It also includes voice command integration with Spotify. You can control volume, media playback, and even search and play a song on Spotify using your voice.

## 🚀 Features

- 🔊 Volume up / down using hand gestures  
- ⏯️ Play / pause / next / previous track  
- 🗣️ Voice-controlled song search and playback via Spotify  
- 🎙️ Mute and unmute microphone  
- 🖼️ Visual interface with intuitive icons  
- 🧠 Uses real-time hand tracking with MediaPipe  
- ⚙️ Sends native media key events to your system (Windows)

## 🧪 Technologies Used

- Python 3
- OpenCV
- MediaPipe
- Pycaw (Windows audio control)
- PyAutoGUI
- Spotipy (Spotify API)
- SpeechRecognition (with Google Speech API)
- dotenv (for managing secrets)
- comtypes (for Windows COM interaction)

## 📦 Requirements

Install the dependencies using the following command:

```bash
pip install -r requirements.txt
