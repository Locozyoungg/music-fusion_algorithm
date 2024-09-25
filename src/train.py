from rl_agent import RLAgent, reward_function
from cvae_model import MusicCVAE
from genre_embeddings import GenreEmbedding
from data_loader import MusicDataset, feature_extractor

# Initialize model, genre embeddings, and RL agent
input_dim = 1024
latent_dim = 64
genre_dim = 10
num_genres = 3  # e.g., jazz, classical, hiphop

cvae_model = MusicCVAE(input_dim, latent_dim, genre_dim)
genre_embedding = GenreEmbedding(num_genres, genre_dim)
rl_agent = RLAgent(cvae_model, reward_function)

# Training loop with RL
for epoch in range(epochs):
    epoch_loss = 0
    for batch in dataloader:
        genre_labels = torch.randint(0, num_genres, (batch_size,))
        genre_embed = genre_embedding(genre_labels)
        loss = rl_agent.train_step(batch, genre_embed)
        epoch_loss += loss

    print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader)}')

    # Save model checkpoint
    if (epoch + 1) % 10 == 0:
        torch.save(cvae_model.state_dict(), f'results/model_epoch_{epoch + 1}.pth')
