import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

class MusicDataset(Dataset):
    def __init__(self, data_dir, genre_labels):
        self.data_dir = data_dir
        self.genre_labels = genre_labels
        self.audio_files = []
        self.labels = []
        
        for genre, label in genre_labels.items():
            genre_dir = os.path.join(data_dir, genre)
            for file in os.listdir(genre_dir):
                if file.endswith('.wav'):  # Assuming WAV files
                    self.audio_files.append(os.path.join(genre_dir, file))
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        
        # Load audio and extract features
        y, sr = librosa.load(audio_path)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc = np.mean(mfcc.T, axis=0)  # Take mean across time axis
        
        return torch.tensor(mfcc, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Example usage
data_dir = 'data/'
genre_labels = {'jazz': 0, 'classical': 1, 'hiphop': 2}
music_dataset = MusicDataset(data_dir, genre_labels)
