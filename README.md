# 🔥 Transformer Playground

A modular, well-documented implementation of the original **Transformer** architecture from [Attention is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762), built using **PyTorch**.

Developed by **xiyuan.chen**, this project is designed for educational purposes and experimentation with different attention mechanisms and transformer variants.

---

## 📌 Features

- ✅ Encoder-Decoder Transformer (from scratch)
- 🧠 Multi-Head Attention (with mask support)
- 🏗️ Positional Encoding (sinusoidal)
- 📈 Teacher Forcing Training Mode
- 📊 BLEU Score Evaluation
- 🔁 Custom Dataset Support (Text-to-Text)
- 🔍 Visualization: Attention Weights with `matplotlib`
- 🚀 Transformer Variants: Add & compare `GPT`, `BERT`, `Linformer`, etc. (WIP)

---

## 📦 Requirements

This project uses the following dependencies:

```bash
pip install torch torchvision
pip install numpy matplotlib
pip install tqdm scikit-learn
pip install nltk
```

##🚀 Quick Start

from transformer.model import Transformer
from transformer.utils import train_model, evaluate_model

model = Transformer(
    src_vocab_size=8000,
    tgt_vocab_size=8000,
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1
)

train_model(model, dataloader, optimizer, loss_fn, num_epochs=10)
evaluate_model(model, test_loader)

##🧠 Architecture Overview

Input Embedding → Positional Encoding
   
    ↓
Multi-Head Attention → Add & Norm

      ↓
Feed Forward → Add & Norm

      ↓
       ↓           ← Skip Connection →
       
Decoder Embedding → Positional Encoding

      ↓
Masked Multi-Head Attention → Add & Norm

      ↓
Encoder-Decoder Attention → Add & Norm

      ↓
Feed Forward → Add & Norm

      ↓
Linear + Softmax

##📚 Training Configuration

python train.py --dataset data/eng-fra.txt --epochs 20 --batch_size 64 --lr 0.0005

##📁 Folder Structure

transformer/

├── model.py         # Transformer model

├── layers.py        # Attention and Feedforward layers

├── utils.py         # Training helpers and evaluation

├── train.py         # Training entry point

├── evaluate.py      # Evaluation script

├── data/            # Raw and processed datasets

└── checkpoints/     # Saved models


##📖 References
Vaswani, A., et al. (2017). Attention is All You Need.

The Annotated Transformer: https://nlp.seas.harvard.edu/2018/04/03/attention.html

PyTorch Tutorials: https://pytorch.org/tutorials/

👤 Author

xiyuan.chen

