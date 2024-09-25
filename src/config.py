# General Config
DATA_DIR = 'data/'  # Path to the music data
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
CHECKPOINT_DIR = 'results/'
NUM_GENRES = 3  # Number of genres in your dataset
GENRE_DIM = 10  # Size of genre embeddings
LATENT_DIM = 64  # Latent space dimension

# Feature extraction
MFCC_DIM = 13  # Number of MFCC coefficients to extract
SPECTROGRAM_DIM = 128  # Dimension of spectrogram features

# Model architecture
ENCODER_HIDDEN_SIZE = 512
DECODER_HIDDEN_SIZE = 256

# Reinforcement learning
RL_LEARNING_RATE = 0.0001
REWARD_FACTOR = 1.0  # Used for adjusting the impact of rewards
