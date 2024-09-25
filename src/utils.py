import numpy as np

def normalize_data(data, min_val=0.0, max_val=1.0):
    """Normalize data to a range between min_val and max_val."""
    min_data, max_data = np.min(data), np.max(data)
    return (data - min_data) / (max_data - min_data) * (max_val - min_val) + min_val

def genre_blending(genre_1, genre_2, alpha=0.5):
    """
    Blend two genres based on a weight factor alpha.
    alpha=0.5 would mean a 50/50 blend of both genres.
    """
    return (alpha * genre_1 + (1 - alpha) * genre_2)

def save_model_checkpoint(model, optimizer, epoch, file_path):
    """Save model checkpoint for resuming training later."""
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, file_path)

def load_model_checkpoint(model, optimizer, file_path):
    """Load model checkpoint and resume training."""
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch
