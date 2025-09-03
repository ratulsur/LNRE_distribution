# LNRE_distribution

# LexiTail – Vocabulary Growth & Rare Event Analysis

LexiTail is a text analytics project designed to explore **vocabulary growth**, **rare event detection**, and **heavy-tail behavior** in textual datasets. It provides tools to preprocess raw text, model vocabulary expansion, validate Zipf/power-law distributions, and analyze rare tokens using **G/Q functions**. The project generates intuitive visualizations (growth curves, rank–frequency plots, word clouds) for research, NLP, and business intelligence storytelling.

---

## 🚀 Features
- **Text Cleaning & Preprocessing**
  - HTML/URL removal, punctuation/number filtering, lemmatization, stopword removal.
- **Vocabulary Growth Modeling**
  - Heaps’ Law (LNRE), simulated LSTM vocabulary growth, BERT-style subword saturation.
- **Rare Event Analysis**
  - Identify low-frequency tokens, extract contexts, map to feedback, and cluster with TF-IDF + KMeans.
- **Heavy-Tail & Distribution Checks**
  - Rank–frequency (Zipf) plots, power-law fitting, tail index estimation.
- **G & Q Functions**
  - Analyze token contributions above/below thresholds; visualize trends over time and across datasets.
- **Visualization Suite**
  - Growth curves, tail contributions, word clouds, log–log plots, and convergence analysis.

---

## 📊 Example Outputs
- **Vocabulary Growth Comparison**
  ![Vocabulary Growth](vocabulary_growth.png)

- **Rank–Frequency Plot**
  ![Zipf Distribution](zipf_fit.png)

- **Rare Word Clusters (Word Cloud)**
  ![Word Cloud](wordcloud.png)

---

## 🛠️ Tech Stack
- **Python 3.10+**
- **Libraries**:  
  - `numpy`, `pandas`, `matplotlib`, `scipy`, `nltk`, `gensim`,  
  - `scikit-learn`, `wordcloud`, `bs4`, `powerlaw`

---

## 📂 Project Structure
