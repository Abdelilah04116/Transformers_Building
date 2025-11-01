from configurator import Config
from tokenizer import CharTokenizer
from dataset import TextDataset
from dataloader import create_dataloader
from model.transformer import GPT
from optimizer.adamw import create_optimizer
from engine import train_one_epoch
import torch, torch.nn as nn
cfg = Config()
text = open(cfg.data_path, "r", encoding="utf-8").read()
tok = CharTokenizer(text)
data = tok.encode(text)
train_data = data[:int(0.9*len(data))]
train_ds = TextDataset(train_data, cfg.block_size)
train_dl = create_dataloader(train_ds, cfg.batch_size)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT(tok.vocab_size, cfg.block_size, cfg.embed_dim, cfg.num_layers, cfg.num_heads, cfg.ff_hidden_dim, cfg.dropout).to(device)
optimizer = create_optimizer(model, cfg.lr)
criterion = nn.CrossEntropyLoss()
for epoch in range(cfg.epochs):
    loss = train_one_epoch(model, train_dl, optimizer, criterion, device)
    print(f"Epoch {epoch} | loss: {loss:.4f}")
    if epoch % cfg.checkpoint_every == 0:
        from checkpoint_manager import save_checkpoint
        save_checkpoint(model, optimizer, epoch)
