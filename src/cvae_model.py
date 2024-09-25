import torch
import torch.nn as nn

class MusicCVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, genre_dim):
        super(MusicCVAE, self).__init__()
        
        # Encoder layers with genre embedding
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + genre_dim, 512),  # Include genre embedding
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Latent space with mean and variance
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Attention mechanism for better fusion
        self.attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=4)
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + genre_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # Outputs between 0 and 1
        )
    
    def encode(self, x, genre_embedding):
        # Concatenate input and genre embedding before encoding
        x = torch.cat([x, genre_embedding], dim=-1)
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, genre_embedding):
        # Apply attention mechanism on latent space
        z = z.unsqueeze(0)  # Add batch dimension for attention
        z, _ = self.attention(z, z, z)
        z = z.squeeze(0)
        z = torch.cat([z, genre_embedding], dim=-1)
        return self.decoder(z)

    def forward(self, x, genre_embedding):
        mu, logvar = self.encode(x, genre_embedding)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z, genre_embedding)
        return reconstructed, mu, logvar

    def loss_function(self, reconstructed, x, mu, logvar):
        recon_loss = nn.functional.mse_loss(reconstructed, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld_loss
