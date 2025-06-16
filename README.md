
# Sentiment Analysis using LSTM

This project implements a Long Short-Term Memory (LSTM) neural network to classify user reviews into sentiment categories (e.g., positive, negative). It includes exploratory data analysis, preprocessing, and training a deep learning model using PyTorch.

---

## 📌 Objectives

- Perform exploratory analysis on a text sentiment dataset.
- Clean, tokenize, and pad textual data.
- Build and train an LSTM model for sentiment classification.
- Visualize word distributions and model performance.
- Evaluate predictions using metrics and sample texts.

---

## 📁 Project Structure

```
lstm-sentiment-analysis/
├── Sentiment analysis using LSTM.ipynb
├── README.md
└── requirements.txt
```

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/lstm-sentiment-analysis.git
cd lstm-sentiment-analysis
pip install -r requirements.txt
```

### 2. Launch the Notebook

```bash
jupyter notebook "Sentiment analysis using LSTM.ipynb"
```

---

## 📊 Technologies Used

- **Python 3**
- **PyTorch** – LSTM model implementation
- **NLTK** – Text preprocessing and tokenization
- **Pandas & NumPy** – Data handling
- **Matplotlib & Seaborn** – Visualization
- **Scikit-learn** – Evaluation metrics
- **WordCloud** – Word frequency visualization

---

## 📈 Model Workflow

1. **Data Exploration**
   - Statistics on review lengths
   - Word cloud generation for sentiment classes

2. **Preprocessing**
   - Tokenization and padding
   - Train-test split

3. **Model Training**
   - LSTM with embedding layer
   - Evaluation using accuracy and loss plots

4. **Evaluation**
   - Class-wise performance
   - Sample prediction outputs

---

## ✅ Sample Output

- Accuracy and loss curves
- Word clouds for sentiment visualization
- Model predictions on user review texts

---

## 🔄 Future Work

- Try bidirectional LSTM or GRU
- Use pre-trained embeddings (e.g., GloVe)
- Extend to multi-class sentiment or emotion detection

---

## 🙌 Acknowledgements

- Inspired by common NLP benchmarks in sentiment analysis
- Based on PyTorch and NLTK tutorials for deep learning with text
