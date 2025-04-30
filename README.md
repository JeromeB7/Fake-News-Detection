# Fake-News-Detection
# 📰 Fake News Detection Using RNN + LSTM with GloVe Embeddings

## 📌 Overview
This project uses a deep learning model to classify political statements as *true* or *false* based on the **LIAR dataset**. It combines pretrained **GloVe embeddings**, **SimpleRNN**, and **LSTM** layers for effective sequence modeling, and handles class imbalance using **SMOTE**.

---

## 🧠 Model Architecture
1. **Embedding Layer** – Converts input text into 300-dimensional GloVe vectors (not trainable).
2. **SMOTE** – Synthetic oversampling applied to balance true/false classes.
3. **SimpleRNN (128 units)** – Captures short-term patterns in text.
4. **LSTM (128 units)** – Learns long-term dependencies.
5. **Dense Layer (64 units)** – Projects features into a compressed abstract space.
6. **Output Layer (1 unit, Sigmoid)** – Predicts binary class: true (1) or false (0).

---

## 🗒️ Why This Model?
- **GloVe embeddings** provide strong semantic meaning out-of-the-box.
- Combining **SimpleRNN and LSTM** captures both short- and long-term text patterns.
- **SMOTE + class weighting** improves generalization on imbalanced data.
- Lightweight yet effective for binary text classification.

---

## 📂 Files
- `train.tsv`, `valid.tsv`, `test.tsv`: Input datasets.
- `glove.6B.300d.txt`: Pretrained embeddings.
- `best_model.keras`: Saved model checkpoint.
- `metrics.txt`, `confusion_matrix.png`, `loss_curves.png`: Evaluation outputs.

