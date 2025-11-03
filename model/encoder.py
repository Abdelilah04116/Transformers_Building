import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encoder amélioré pour la classification de sentiment avec architecture plus profonde.
    """
    def __init__(self, vocab_size=128, embed_dim=128, num_classes=3, num_layers=2, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Couches conv1D pour capturer les patterns locaux
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        # Couches linéaires avec dropout
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, y=None):
        # x : (batch, seq_len) (indices)
        # y : (batch,) (labels)
        emb = self.embedding(x)  # (batch, seq_len, embed_dim)
        
        # Convolution 1D (nécessite: batch, channels, seq_len)
        emb = emb.transpose(1, 2)  # (batch, embed_dim, seq_len)
        x = F.relu(self.conv1(emb))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        
        # Pooling global (moyenne sur la séquence)
        x = x.mean(dim=2)  # (batch, hidden_dim)
        x = self.layer_norm(x)
        
        # Couches linéaires avec activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)  # (batch, num_classes)

        loss = None
        if y is not None:
            loss = nn.CrossEntropyLoss()(logits, y)
        return logits, loss
