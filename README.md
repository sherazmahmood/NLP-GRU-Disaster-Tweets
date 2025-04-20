# NLP Disaster Tweet Classification — GRU Model

Welcome to the repository for my project on predicting disaster-related tweets using deep learning (GRU-based models).

This project was completed as part of the **University of Colorado Boulder — DTSA 5511: Introduction to Deep Learning** course.

The project is based on the Kaggle competition ["NLP with Disaster Tweets"](https://www.kaggle.com/c/nlp-getting-started).

## Project Overview

In this project, we aim to classify whether a tweet is about a real disaster or not, based on its text content.  
We explored sequential neural networks (GRUs) enhanced with pretrained word embeddings (GloVe), simple feature engineering, and careful model tuning.


## Repository Structure

| File/Folder | Description |
|:---|:---|
| `NLP_GRU_Disaster_tweets.ipynb` | Main Jupyter Notebook — Full workflow: data cleaning, EDA, model building, training, and evaluation |
| `saved_models/` | Folder containing trained model `.h5` files |
| `submission.csv` | Example Kaggle submission file |
| `README.md` | This document |


## How to Run

This project was developed and trained using **Google Colab** with **TPU acceleration**.

1. Open the notebook `NLP_GRU_Disaster_tweets.ipynb` in Colab.
2. Upload the Kaggle competition dataset (`train.csv`, `test.csv`).
3. Install required libraries:
    ```bash
    pip install nltk gensim
    ```
4. Set hardware accelerator to **TPU** (via Colab: Runtime → Change Runtime Type → TPU).
5. Run all cells.

Training time is fast (~5-10 minutes depending on TPU and model size).

## Highlights of the Approach

- **Exploratory Data Analysis (EDA):**
  - Text cleaning, tokenization, stopword removal.
  - Word frequency analysis for both disaster and non-disaster tweets.
  - Emergency and non-emergency keyword feature engineering.

- **Word Embedding Strategy:**
  - Used **pretrained GloVe Twitter embeddings** for better word vector representations.
  - Experiments with smaller embedding sizes (50, 100, 150 dimensions) for efficiency.

- **Model Architecture:**
  - **Bidirectional GRU** to capture sequential context from both directions.
  - **Dropout regularization** to prevent overfitting.
  - **Auxiliary inputs** for emergency and non-emergency keyword indicators.
  - Final **Dense sigmoid** layer for binary classification.

- **Training Optimization:**
  - Early stopping based on F1 score.
  - Learning rate scheduling on plateau.

- **Performance:**
  - Achieved an F1 score of **~0.777** on validation set using GRU + GloVe embeddings.
  - Strong baseline performance without transformer architectures (e.g., BERT).

## Results

| Experiment | F1 Score |
|:---|:---|
| Word2Vec trained from scratch | ~0.55–0.60 |
| GloVe embeddings + GRU | ~0.77–0.78 |

Switching to pretrained embeddings and optimizing model architecture dramatically improved performance.

## ✏Key Learnings

- **Pretrained embeddings** (GloVe) are critical when working with small NLP datasets.
- **Simpler architectures** (e.g., 1-layer GRU) perform better than deeper or larger models on small, noisy datasets.
- **Feature engineering** (emergency/non-emergency keywords) provides valuable structured information beyond pure text.

## References

- Pennington, J., Socher, R., & Manning, C. (2014). GloVe: Global Vectors for Word Representation. [EMNLP 2014](https://aclanthology.org/D14-1162/).
- Kwan, M. (2019). Finding the optimal number of dimensions for word embeddings. [Medium article](https://medium.com/@matti.kwan/finding-the-optimal-number-of-dimensions-for-word-embeddings-f19f71666723).
- Kaggle. (2020). NLP with Disaster Tweets Challenge. [Kaggle Competition Link](https://www.kaggle.com/c/nlp-getting-started).

## Acknowledgments

- Special thanks to the Kaggle community for discussions and tutorials that helped guide parts of the model design.
- Built with TensorFlow, Keras, Gensim, and NLTK libraries.
