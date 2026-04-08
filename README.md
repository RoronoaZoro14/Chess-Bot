# ♟️ Chess Bot (CNN + MCTS)

A deep learning–based chess engine that predicts moves using a **Convolutional Neural Network (CNN)** and optionally improves decision-making using **Monte Carlo Tree Search (MCTS)**.

---

## 🚀 Overview

This project implements a chess-playing AI that:

- Learns board representations using CNN
- Predicts moves from real game datasets
- Enhances gameplay with search-based optimization (MCTS)
- Compares performance between pure neural network vs hybrid approach

The goal is to explore how deep learning and classical search algorithms can be combined for intelligent decision-making in chess.

---

## 🧠 Features

- Board representation as tensors (12 × 8 × 8)
- Move prediction using CNN
- Residual blocks (ResNet-style architecture)
- Training on large-scale chess datasets
- Optional Monte Carlo Tree Search (MCTS)
- GPU acceleration support (CUDA)
- Training & validation pipeline
- Model evaluation and comparison

---

## 🏗️ Project Structure

```
Chess-Bot/
│── data/                  # Preprocessed datasets (boards, moves)
│── models/                # CNN architecture & training scripts
│── preprocess.py          # Converts raw dataset → tensors
│── train.py               # Model training script
│── mcts.py                # Monte Carlo Tree Search (optional)
│── utils.py               # Helper functions
│── requirements.txt       # Dependencies
│── README.md              # Project documentation
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/RoronoaZoro14/Chess-Bot.git
cd Chess-Bot
```

### 2. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

- Uses chess game datasets (e.g., PGN or CSV format)
- Preprocessing converts:
  - Board → tensor representation
  - Moves → indexed labels

Run preprocessing:
```bash
python preprocess.py
```

---

## 🏋️ Training

Train the CNN model:

```bash
python train.py
```

Training includes:
- Train/validation split
- Loss tracking
- Accuracy evaluation
- Model checkpoint saving

---

## 🌲 MCTS Integration (Optional)

To improve move selection:

- Use CNN for policy prediction
- Apply MCTS for deeper search

```bash
python mcts.py
```

---

## 📈 Model Architecture

- Input: `12 × 8 × 8` board tensor  
- Multiple convolution layers  
- Residual blocks:
```python
output = ReLU(x + F(x))
```
- Fully connected layer → move probabilities

---

## 🧪 Results

| Model Type | Description |
|------------|-------------|
| CNN Only   | Fast, learned patterns |
| CNN + MCTS | Stronger, search-enhanced decisions |

---

## 🎯 Future Improvements

- Add reinforcement learning (self-play)
- Integrate opening databases
- Optimize inference speed
- Deploy as web or API service
- Play against external engines (e.g., Stockfish)

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo  
2. Create a new branch  
3. Commit your changes  
4. Open a pull request  

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 👤 Author

Megh Patel  
GitHub: https://github.com/RoronoaZoro14  

---

## ⭐ Acknowledgements

- Inspired by deep learning approaches to chess
- Based on CNN architectures and search algorithms used in modern AI systems
