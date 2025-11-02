# Projet Transformer - Documentation Technique

Ce dépôt contient une implémentation pédagogique et compacte d’un modèle Transformer de type GPT (génération de texte caractère à caractère). L’objectif est de faciliter l’expérience utilisateur, l’entraînement, l’inférence, la configuration et la reproductibilité du modèle, tout en gardant une séparation claire des responsabilités dans chaque module.

## Installation
1. Installer les dépendances :
```bash
pip install -r requirements.txt
```
2. Préparer vos données dans `data/input.txt`.
3. Adapter si besoin la configuration dans `config.yaml`.

## Structure du projet

- `train.py` : Script principal d’entraînement du modèle.
- `inference.py` : Script d’inférence/génération de texte à partir d’un prompt.
- `configurator.py` : Charge la configuration à partir d’un fichier YAML.
- `config.yaml` : Fichier des hyperparamètres et chemins.
- `tokenizer.py` : Tokenizer caractère à caractère.
- `dataset.py` : Dataset Torch pour la donnée séquencée.
- `dataloader.py` : Générateur de DataLoader PyTorch.
- `engine.py` : Boucle d'entraînement principale (une époque).
- `checkpoint_manager.py` : Sauvegarde/chargement des checkpoints.
- `model/` : Module du modèle Transformer (attention, feedforward, bloc, GPT).
- `optimizer/` : Optimiseur AdamW pour l'entraînement.
- `core_eval.py`, `loss_eval.py`, `common.py`, `execution.py`, `report.py` : utilitaires d’évaluation, logging, seed, etc.
- `ui.html` : Interface utilisateur web simple pour tests.
- `requirements.txt` : Dépendances Python.

## Explications détaillées par module

### 1. Entraînement (`train.py`)
- **Imports principaux :** config, tokenizer, dataset, modèle, optimiser, etc.
- Chargement du texte, tokenisation, création du dataset découpé, DataLoader, etc.
- Instanciation du modèle GPT avec paramètres de config.
- Boucle d’entraînement avec sauvegarde régulière du modèle.

### 2. Inférence (`inference.py`)
- Recharge la config, le texte et le tokenizer, puis le modèle.
- Charge un checkpoint existant.
- Fonction `generate()` : génère une séquence à partir d’un prompt (softmax, multinomial sampling).

### 3. Configuration (`configurator.py`, `config.yaml`)
- Classe `Config` : charge automatiquement tout le YAML dans les attributs de l'objet.
- Le yaml gère les chemins, dimensions de l’embed, batch size, etc.

### 4. Tokenisation et données
- **`tokenizer.py > CharTokenizer`** :
  - Convertit du texte en indices (et inversement) pour chaque caractère distinct du corpus.
  - `encode(s)`: transforme la chaîne en indices.
  - `decode(list)`: reconstruit le texte à partir de la liste d’indices.
- **`dataset.py > TextDataset`** :
  - Permet d’itérer sur des blocs de texte pour prediction séquence.
  - `__getitem__`: Retourne (entrée, target) pour le bloc courant.
- **`dataloader.py > create_dataloader`** :
  - Fournit un DataLoader torch standard, avec shuffle.

### 5. Modèle Transformer
- **`model/attention.py > MultiHeadSelfAttention`** :
  - Implémente l’attention multi-tête avec masquage causal.
- **`model/feedforward.py > FeedForward`** :
  - Dense + activation GELU entre chaque block de l’architecture Transformer.
- **`model/transformer.py`** :
  - `TransformerBlock` : Block standard (Attention + Add&Norm + FeedForward + Add&Norm).
  - `GPT` : Embedding token/positions, empilement de blocks, normalisation finale puis projection logits sur le vocabulaire.
  - `forward(idx)`: Propagation avant sur séquence de tokens.
- **`model/__init__.py`** : Import de la classe GPT du module principal.

### 6. Optimiseur et Checkpoints
- **`optimizer/adamw.py > create_optimizer`** : Retourne l’optimiseur AdamW paramétré.
- **`checkpoint_manager.py`** :
  - `save_checkpoint`: sauvegarde état modèle+optimiseur+epoch.
  - `load_checkpoint`: recharge un checkpoint.

### 7. Utilitaires et autres modules
- **`common.py`** :
  - `set_seed(seed)`: force la reproductibilité.
  - `count_parameters(model)`: compte les paramètres entraînables du modèle.
- **`core_eval.py`** :
  - `ppl_from_loss(loss)`: calcule la perplexité à partir de la loss.
- **`loss_eval.py`** :
  - `get_criterion()`: retourne la CrossEntropyLoss.
- **`execution.py`** :
  - `print_startup(cfg)`: loggue le contenu de la config.
- **`report.py`** :
  - `simple_report(epoch, loss)`: loggue la valeur de la loss à chaque epoch.

### 8. Interface web (`ui.html`)
- Un simple frontend pour générer des séquences en ligne (optionnel, à compléter selon besoins).

### 9. Environnement
- `requirements.txt` liste torch, tqdm, pyyaml et autres utilitaires.
- `venv/` contient l’environnement virtuel local Python (à ne pas versionner).

## Exemple d’utilisation

**Entraîner le modèle :**
```bash
python train.py
```

**Générer du texte :**
```bash
python inference.py
```

L’ensemble du code est découpé pour une compréhension immédiate de chaque brique, dans une optique pédagogique et reproductible. Adapter les paramètres dans `config.yaml` pour toute expérimentation.

---

*Documentation générée pour la clarté et la prise en main de ce projet Transformer minimalist.*

-------------------------------
Realiser Par Ourti Abdelilah 

