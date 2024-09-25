import librosa
import numpy as np

def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)  # Mean pooling across time

def extract_chroma(file_path):
    y, sr = librosa.load(file_path)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return np.mean(chroma.T, axis=0)

def extract_spectrogram(file_path):
    y, sr = librosa.load(file_path)
    spec = librosa.feature.melspectrogram(y=y, sr=sr)
    return np.mean(spec.T, axis=0)

def extract_features(file_path):
    mfcc = extract_mfcc(file_path)
    chroma = extract_chroma(file_path)
    spectrogram = extract_spectrogram(file_path)
    
    return np.hstack([mfcc, chroma, spectrogram])  # Concatenate all features
