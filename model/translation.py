import torch
import torch.nn as nn
import torch.nn.functional as F

class TranslationModel(nn.Module):
    """
    Modèle de traduction amélioré avec architecture plus profonde.
    """
    def __init__(self, vocab_src, vocab_tgt, embed_dim=128, block_size=32, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_src, embed_dim)
        
        # Couches pour encoder la séquence source
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        # Couches linéaires pour générer les tokens cibles
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, vocab_tgt)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.block_size = block_size
        self.vocab_tgt = vocab_tgt

    def forward(self, src, tgt=None):
        # src: (batch, seq_len), tgt: (batch, seq_len)
        emb = self.embedding(src)  # (batch, seq_len, embed_dim)
        
        # Convolution 1D
        emb = emb.transpose(1, 2)  # (batch, embed_dim, seq_len)
        x = F.relu(self.conv1(emb))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        
        # Pooling global
        x = x.mean(dim=2)  # (batch, hidden_dim)
        x = self.layer_norm(x)
        
        # Couches linéaires
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)  # (batch, vocab_tgt)

        loss = None
        if tgt is not None:
            # Calculer la loss sur le premier token cible (version simplifiée)
            loss = nn.CrossEntropyLoss()(logits, tgt[:,0])
        return logits, loss
    
    @torch.no_grad()
    def generate(self, src, max_length=None):
        """
        Génère une séquence de traduction à partir de src.
        Pour un modèle simple, on génère token par token.
        """
        self.eval()
        if max_length is None:
            max_length = self.block_size
        
        # Encodage source (même processus que forward)
        emb = self.embedding(src)  # (batch, seq_len, embed_dim)
        
        # Convolution 1D
        emb = emb.transpose(1, 2)  # (batch, embed_dim, seq_len)
        x = F.relu(self.conv1(emb))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        
        # Pooling global
        x = x.mean(dim=2)  # (batch, hidden_dim)
        x = self.layer_norm(x)
        
        # Génération séquentielle simple (token par token)
        # On utilise la même représentation encodée pour chaque token
        generated_tokens = []
        for _ in range(min(max_length, 50)):  # Limite à 50 tokens
            # Passer par les couches linéaires
            h = F.relu(self.fc1(x))
            h = self.dropout(h)
            h = F.relu(self.fc2(h))
            h = self.dropout(h)
            logits = self.fc3(h)  # (batch, vocab_tgt)
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            generated_tokens.append(next_token.item())
            
            # Stop si token de fin (0 généralement)
            if next_token.item() == 0:
                break
        
        return generated_tokens
