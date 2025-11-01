import torch
def generate(model, start_tokens, max_new_tokens, tokenizer, device="cpu"):
    model.eval()
    idx = torch.tensor([start_tokens], dtype=torch.long).to(device)
    for _ in range(max_new_tokens):
        logits = model(idx)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return tokenizer.decode(idx[0].tolist())
