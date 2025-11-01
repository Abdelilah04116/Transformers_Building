import torch
def create_optimizer(model, lr):
    return torch.optim.AdamW(model.parameters(), lr=lr)
