import torch
import torch.nn as nn

class GenreEmbedding(nn.Module):
    def __init__(self, num_genres, genre_dim):
        super(GenreEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_genres, genre_dim)
    
    def forward(self, genre_labels):
        return self.embedding(genre_labels)

# Example usage:
# jazz = 0, classical = 1, hiphop = 2
genre_labels = torch.tensor([0, 1, 2])  # One-hot or integer encoding
genre_dim = 10
embedding = GenreEmbedding(num_genres=3, genre_dim=genre_dim)
genre_embeddings = embedding(genre_labels)
