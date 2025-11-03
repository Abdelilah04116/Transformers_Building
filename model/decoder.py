import torch
import torch.nn as nn

class Decoder(nn.Module):
    """
    Decoder pour la génération de texte en fonction d'un nombre de mots donné.
    Version debug : contient juste une couche linéaire pour débloquer l'entraînement.
    """
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)  # Couche fictive

    def forward(self, x, y=None, max_length=20):
        logits = self.linear(torch.zeros((1,10)))
        loss = torch.zeros(1, requires_grad=True, device=logits.device)
        return logits, loss
