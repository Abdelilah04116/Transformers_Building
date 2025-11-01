Transformer Project (minimal)
=============================
Files:
  - train.py         : training script
  - inference.py     : basic generation helper
  - model/           : transformer model implementation
  - tokenizer.py     : char-level tokenizer
  - dataset.py       : dataset class
  - dataloader.py    : dataloader helper
  - engine.py        : training loop
  - optimizer/adamw  : optimizer factory
  - checkpoint_manager.py
  - config.yaml
How to run:
  1) pip install -r requirements.txt  (torch, tqdm, pyyaml)
  2) edit config.yaml paths/hyperparams
  3) python train.py
