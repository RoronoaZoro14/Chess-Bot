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
│── chess_games.csv/                        # Preprocess dataset for CNN, outputs (boards, moves)
│── lichess_puzzles.csv                     # Evaluation dataset for CNN & CNN + MCTS for testing on puzzles (Big file not available on Github)
│── Train_from_preprocessing.py/            # CNN architecture & training scripts for RAW CNN model, outputs chess_cnn.pt
│── chess_cnn.pt                            # Trained model using RAW CNN
│── mcts.py                                 # Monte Carlo Tree Search (optional)
│── Train_form_preprocessing_MCTS.py        # Trains the model with MCTS, outputs chess_cnn_mcts.pt
│── evaluate.py                             # Evalutes the CNN model based on inputs min_elo and max_games
│── evaluate_mcts.py                        # Evalutes the CNN model + MCTS based on inputs min_elo and max_games
│── chess_gui.py                            # A GUI interface to interact with the bot
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


---

## 📊 Dataset

- Uses chess game datasets (e.g., PGN or CSV format)
- Preprocessing converts:
  - Board → tensor representation
  - Moves → indexed labels

Run preprocessing:
```bash
python Preprocessing.py
```

---

## 🏋️ Training

Train the CNN model:

```bash
python Train_from_preprocessing.py
```

Training includes:
- Train/validation split
- Loss tracking
- Accuracy evaluation
- Model checkpoint saving

---

## 🌲 Evaluation

Evaluate the model:
 - First download the 'lichess_puzzles.csv' dataset from this link: https://drive.google.com/file/d/11fJAHGgm55gd4C38su5QHHDFO_7HrCTa/view?usp=drive_link
 - Add it to project folder

```bash
python evalauate.py
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
