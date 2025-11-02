import torch
from model.transformer import GPT
from tokenizer import CharTokenizer
from configurator import Config
import os

# Charger config
cfg = Config("config.yaml")

# Charger text pour tokenizer
text = open(cfg.data_path, "r", encoding="utf-8").read()

# Charger tokenizer
tokenizer = CharTokenizer(text)
vocab_size = tokenizer.vocab_size

# Choisir device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Charger modèle GPT
model = GPT(
    vocab_size=vocab_size,
    block_size=cfg.block_size,
    embed_dim=cfg.embed_dim,
    num_layers=cfg.num_layers,
    num_heads=cfg.num_heads,
    ff_hidden_dim=cfg.ff_hidden_dim,
    dropout=cfg.dropout
).to(device)

# Charger checkpoint
checkpoint_path = "checkpoints/ckpt.pt"
assert os.path.exists(checkpoint_path), "❌ ERREUR: checkpoint introuvable !"

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()

print("✅ Modèle + checkpoint chargé avec succès")

def generate(model, start_tokens, max_new_tokens, tokenizer, device):
    model.eval()
    idx = torch.tensor(start_tokens, dtype=torch.long).unsqueeze(0).to(device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -cfg.block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return tokenizer.decode(idx[0].tolist())

# Prompt
prompt = "hello"
start_tokens = tokenizer.encode(prompt)

# Génération
output = generate(
    model=model,
    start_tokens=start_tokens,
    max_new_tokens=100,
    tokenizer=tokenizer,
    device=device
)

print("\n===== OUTPUT =====")
print(output)
