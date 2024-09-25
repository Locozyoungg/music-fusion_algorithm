# Music Fusion AI using Conditional VAE

This repository contains an advanced AI model that generates music by fusing multiple genres into a unique sound using a Conditional Variational Autoencoder (CVAE) and Reinforcement Learning.

## Structure

- **src/**: Source code for model, data handling, and feature extraction.
  - `data_loader.py`: Load and preprocess the music dataset.
  - `feature_extraction.py`: Extract audio features (MFCC, chroma, etc.).
  - `genre_embeddings.py`: Create embeddings for representing genres.
  - `cvae_model.py`: The Conditional VAE model with attention mechanism.
  - `train.py`: Training script for the model.
  - `rl_agent.py`: Reinforcement Learning agent to improve music generation.
  - `utils.py`: Helper functions (e.g., data normalization, model saving).
  - `config.py`: Hyperparameters and configuration settings.

- **notebooks/**: Jupyter notebooks for analysis.
- **results/**: Checkpoints and generated outputs.
- **requirements.txt**: List of dependencies to install.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/music-fusion-ai.git
   cd music-fusion-ai

Install dependencies:
pip install -r requirements.txt

Prepare your dataset: Place your music files into data/ in separate subfolders for each genre (e.g., data/jazz, data/classical).

Run the training:
python src/train.py

Optionally, explore and analyze results in the provided Jupyter notebooks.

Model
This AI model is built using a Conditional Variational Autoencoder (CVAE) with an attention mechanism. It also leverages Reinforcement Learning to fine-tune the quality of the generated music.

Input: Music features (MFCC, chroma, spectrogram) and genre embeddings.
Output: Generated music feature vector conditioned on genre input.
Loss: Combination of reconstruction loss and KL divergence for VAE, with an additional reward mechanism for quality improvement.

License
MIT License


---

### **`requirements.txt`**

Add all the required libraries for the project.

```txt
torch==2.0.0
torchvision==0.15.0
torchaudio==2.0.0
librosa==0.9.2
numpy==1.23.5
matplotlib==3.5.1
jupyter==1.0.0
