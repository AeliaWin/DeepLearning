import streamlit as st
import librosa
from keras.models import load_model
import numpy as np
import moviepy.editor as mp
from pytube import YouTube
import youtube_dl
import yt_dlp
from pydub import AudioSegment
import os
import math 
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotdl
import mutagen
import nltk

model = load_model('my_model.h5')
SPOTIPY_CLIENT_ID = '3603967e0074491db8302632ea86cc83'
SPOTIPY_CLIENT_SECRET = '595ff04c1ba74bd48ffbdd7e6d63a52b'

st.title("Music Genre Classification")
SAMPLE_RATE = 22050
TRACK_DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

# Set up Spotify API client
client_credentials_manager = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


def augment_audio(audio_data):
    # Pitch shifting
    n_steps = np.random.randint(-2, 3)
    audio_data = librosa.effects.pitch_shift(audio_data, sr=SAMPLE_RATE, n_steps=n_steps)  # Shift by 2 half-steps
    # Time stretching
    rate = np.random.uniform(0.9,1.1)
    time_stretch = librosa.effects.time_stretch(audio_data, rate=rate)  # Speed up slightly
    # Add noise
    noise = np.random.randn(len(time_stretch))
    augmented_data = time_stretch + 0.005 * noise  # Adding a small amount of noise
    return augmented_data

import librosa

def extract_metadata_librosa(audio_file):
    try:
        audio = mutagen.File(audio_file)
        metadata = {
            "artist": audio.get('artist',''),
            "title": audio.get('title',''),
            "duration": librosa.get_duration(y=audio_file, sr=SAMPLE_RATE)
        }

        lyrics = audio.get('lyrics','')
        if lyrics:
            detected_language = nltk.detect_languages(lyrics)[0]
            metadata['is_english'] = detected_language == 'english'
        else:
            metadata['is_english'] = False
        return metadata
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return None


def extract_mfcc(audio_file, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=10):
    samples_per_segment = int(len(audio_file) / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
    mfcc_features = []
    max_segments = len(audio_data) // samples_per_segment

    for d in range(min(num_segments,max_segments)):
        start = samples_per_segment * d
        finish = start + samples_per_segment

        if finish > len(audio_data):
            st.warning(f"Segment {d} exceeds audio length.")
            break

        mfcc = librosa.feature.mfcc(y=audio_file[start:finish], sr=SAMPLE_RATE, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T
        # mfcc = mfcc[np.newaxis, ..., np.newaxis]

        # st.write(f"Segment {d}: MFCC shape {mfcc.shape}") 

        if mfcc.shape[0] >= 128:  # Adjust this as necessary
            mfcc = mfcc[:128, :num_mfcc]
            mfcc = np.pad(mfcc, ((0, 128 - mfcc.shape[0]), (0, 0)), mode='constant')  # Pad to get to width of 32
            mfcc = mfcc[..., np.newaxis]  # Add a channel dimension
            mfcc_features.append(mfcc)
    
    if mfcc_features:
        # st.write(f"Extracted {len(mfcc_features)} segments.")
        return np.array(mfcc_features)
    else:
        # st.error("Could not extract MFCC features.")
        return None

def predict_genre(model, audio_data):
    """Predict the genre of an audio file using the loaded model."""
    
    genre_dict = {
        0: "blues",
        1: "classical",
        2: "country",
        3: "disco",
        4: "hiphop",
        5: "jazz",
        6: "metal",
        7: "pop",
        8: "reggae",
        9: "rock",
    }
    augmented_data = augment_audio(audio_data)
    mfcc = extract_mfcc(augmented_data)
    if mfcc is not None:
        # st.write(mfcc.shape)  # Check shape of the extracted MFCC features
        mfcc = mfcc[..., np.newaxis]  # Ensure the last dimension is 1 if needed (shape: (num_segments, 128, 13, 1))
        prediction = model.predict(mfcc)

        avg_prediction = np.mean(prediction, axis=0)

        top_5_indices = np.argsort(avg_prediction)[-5:][::-1]  # Sort probabilities and get the top 5
        top_5_genres = [(genre_dict[i], avg_prediction[i]) for i in top_5_indices]

        st.write("Top 5 Predicted Genres")
        for genre,probability in top_5_genres:
            st.write(f"{genre}: {probability:.2%}")
        
    else:
        st.write("Unknown")

import yt_dlp
import librosa
import nltk

def extract_metadata_from_youtube(youtube_url):
    """Extract song title and uploader name from a YouTube video."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'noplaylist': True,
        'quiet': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)  # No need to download, just extract info
        song_name = info_dict.get('title', 'Unknown Title')  # Extract song title
        uploader_name = info_dict.get('uploader', 'Unknown Uploader')  # Extract uploader name

    return song_name, uploader_name


option = st.selectbox("Choose an input method:", ["Upload File", "Insert YouTube Link","Enter Spotify URL"])
# audio_data,sr = 0

if option == "Upload File":
    uploaded_file = st.file_uploader("Choose an audio file")
    if uploaded_file is not None:
        # Check file format and convert if necessary
        if uploaded_file.name.endswith('.mp4'):
            try:
                video_clip = mp.VideoFileClip(uploaded_file)
                audio_clip = video_clip.audio
                audio_clip.write_audiofile('temp.wav')
                audio_data, sr = librosa.load('temp.wav', sr=22050)
            except Exception as e:
                st.error(f"Error processing MP4 file: {e}")  # Handle potential errors
                # continue  # Skip processing this file if there's an error
        else:
            audio_data, sr = librosa.load(uploaded_file, sr=22050)
            # st.write(len(audio_data))
        predict_genre(model,audio_data)

        
elif option == "Insert YouTube Link":
    youtube_url = st.text_input("Enter YouTube URL:")

    ffmpeg_path = 'C:\Program Files (x86)\FormatFactory\FFModules\Encoder'

    if youtube_url:
        song_name, uploader_name = extract_metadata_from_youtube(youtube_url)
        st.write(f"The insert song link: {uploader_name}_{song_name}")
        ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'ffmpeg_location': ffmpeg_path,
        'outtmpl': 'audio.%(ext)s',
    }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
            audio_data, sr = librosa.load('audio.wav', sr=22050)
        
            predict_genre(model,audio_data)
            os.remove('audio.wav')
            
elif option == "Enter Spotify URL":
    spotify_url = st.text_input("Enter Spotify URL:")
    if spotify_url:
         
            track_id = spotify_url.split("/")[-1].split("?")[0]
            os.system(f"spotdl {spotify_url}")
            mp3_files = [f for f in os.listdir() if f.endswith('.mp3')]
            st.write(mp3_files)

            if not mp3_files:
                st.error("Failed to find the downloaded MP3 file.")

            output_mp3 = mp3_files[0]
            output_wav = f"{output_mp3[:-4]}.wav" 

            st.write(f"The inserted song link from Spotify: {output_mp3[:-4]}")

            audio = AudioSegment.from_mp3(output_mp3)
            audio.export(output_wav, format="wav")
            
            os.remove(output_mp3)
            audio_data, sr = librosa.load(output_wav, sr=22050)
            predict_genre(model,audio_data)
            os.remove(output_wav)


            


            