# Theoritical analysis of Deep Learning Architecture for Natural Language Processing

A comprehensive machine learning project for classifying Twitter airline sentiment into three categories: **Negative**, **Neutral**, and **Positive**. This project implements and compares multiple deep learning architectures (RNN, LSTM, GRU, TextCNN, DNN) and classical machine learning models (Naive Bayes, Random Forest) using hyperparameter optimization.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Data Preprocessing](#data-preprocessing)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Usage](#usage)
- [Requirements](#requirements)
- [Key Findings](#key-findings)

## Overview

This project performs sentiment analysis on Twitter data related to airline customer feedback. The goal is to classify tweets into three sentiment categories using various machine learning approaches. The project includes:

- **15 Deep Learning Models** across 5 architectures (SimpleRNN, LSTM, GRU, TextCNN, DNN)
- **5 Classical ML Models** (Naive Bayes with 3 configurations, Random Forest with 2 configurations)
- Comprehensive hyperparameter optimization with small/medium/large configurations
- Extensive evaluation metrics and visualizations

## Dataset

**Source**: [Kaggle - Twitter US Airline Sentiment Dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)

- **Total Records**: 14,640 tweets
- **Features**: Original dataset contains 15 columns including tweet text, airline name, sentiment, negative reasons, location, etc.
- **Labels**: 3 sentiment classes
  - `0`: Negative (majority class)
  - `1`: Neutral
  - `2`: Positive (minority class)
- **Splits**: 
  - Training: 10,247 samples (70%)
  - Validation: 2,197 samples (15%)
  - Test: 2,196 samples (15%)
- **Stratified**: Yes, to maintain class distribution across splits

### Class Distribution

The dataset is imbalanced, which is handled using balanced class weights during training:
- **Negative**: Majority class
- **Neutral**: Moderate representation
- **Positive**: Minority class (receives highest class weight: ~2.07)

## Methodology

### Data Preprocessing Pipeline

1. **Column Selection & Removal**:
   - **Dropped**: Unique identifiers, missing/sparse columns (tweet_id, tweet_coord, tweet_location, user_timezone, etc.)
   - **Merged**: `negativereason` into `text_clean` to preserve signal
   - **Kept for modeling**: Only `text_clean` (features) and `label` (target)

2. **Text Cleaning**:
   - Convert to lowercase
   - Remove URLs and mentions (@username)
   - Remove special characters (keep only alphanumeric and spaces)
   - Normalize whitespace
   - Merge negative reason text into cleaned text

3. **Feature Engineering**:
   - **For Deep Learning**: Tokenization with max vocabulary of 20,000 words, sequences padded/truncated to length 128
   - **For Classical ML**: TF-IDF vectorization with 20,000 features, 1-2 gram ranges

4. **Data Splitting**: Stratified 70/15/15 split to maintain class distribution

### Column Decisions

| Column | Action | Reason |
|--------|--------|--------|
| `text_clean` | MODEL_INPUT | Preprocessed text — **only input feature** used by models |
| `label` | TARGET | Encoded sentiment (0=neg,1=neu,2=pos) — **only target** |
| `tweet_id` | DROP | Unique identifier — not a model feature |
| `airline_sentiment` | KEEP_MEMORY_ONLY | Used only to create numeric label |
| `negativereason` | MERGE_TO_TEXT | Merged into text_clean to preserve signal |
| `airline`, `retweet_count` | KEEP_MEMORY_ONLY | Metadata features not used in this text-only model |
| `tweet_coord`, `tweet_location`, `user_timezone` | DROP | High missing values (>30%) and noisy |

## Models Implemented

### Deep Learning Models (Keras/TensorFlow)

All DL models use:
- **Embedding Layer**: Converts tokenized sequences to dense vectors
- **SpatialDropout1D**: Regularization (0.2 dropout rate)
- **L2 Regularization**: Applied to recurrent/dense layers (1e-4)
- **Class Weights**: Balanced weights computed from training data
- **Optimizer**: Adam with gradient clipping (clipnorm=1.0)
- **Loss**: Sparse categorical crossentropy
- **Callbacks**: Early stopping (patience=4), ReduceLROnPlateau (patience=2), ModelCheckpoint

#### 1. SimpleRNN
- **Architecture**: Embedding → SpatialDropout → SimpleRNN → Dense → Dropout → Output
- **Configurations**: 
  - Small: embed=64, rnn_units=32, dense=32, lr=2e-4
  - Medium: embed=128, rnn_units=64, dense=64, lr=1e-4
  - Large: embed=256, rnn_units=128, dense=128, lr=1e-4

#### 2. LSTM
- **Architecture**: Embedding → SpatialDropout → LSTM → Dense → Dropout → Output
- **Configurations**: 
  - Small: embed=64, lstm_units=32, dense=32, lr=2e-4
  - Medium: embed=128, lstm_units=64, dense=64, lr=1e-4
  - Large: embed=256, lstm_units=128, dense=128, lr=1e-4

#### 3. GRU
- **Architecture**: Embedding → SpatialDropout → GRU → Dense → Dropout → Output
- **Configurations**: 
  - Small: embed=64, gru_units=32, dense=32, lr=2e-4
  - Medium: embed=128, gru_units=64, dense=64, lr=1e-4
  - Large: embed=256, gru_units=128, dense=128, lr=1e-4

#### 4. TextCNN
- **Architecture**: Embedding → Multiple Conv1D layers with different kernel sizes (3,4,5) → GlobalMaxPooling → Concatenate → Dense → BatchNorm → Dropout → Output
- **Configurations**: 
  - Small: embed=64, filters=64, dense=32, lr=2e-4
  - Medium: embed=128, filters=128, dense=64, lr=1e-4
  - Large: embed=256, filters=256, dense=128, lr=1e-4

#### 5. DNN (Dense Neural Network)
- **Architecture**: Embedding → SpatialDropout → GlobalAveragePooling → Dense → Dropout → Dense → Dropout → Output
- **Configurations**: 
  - Small: embed=64, dense=64, lr=2e-4
  - Medium: embed=128, dense=128, lr=1e-4
  - Large: embed=256, dense=256, lr=5e-5

### Classical Machine Learning Models (Scikit-learn)

#### 1. Multinomial Naive Bayes
- **Features**: TF-IDF vectors (20,000 features, 1-2 grams)
- **Configurations**:
  - `nb_alpha_1.0`: alpha=1.0 (default smoothing)
  - `nb_alpha_0.5`: alpha=0.5 (less smoothing)
  - `nb_alpha_0.1`: alpha=0.1 (minimal smoothing)

#### 2. Random Forest
- **Features**: TF-IDF vectors (20,000 features, 1-2 grams)
- **Configurations**:
  - `rf_small`: 200 estimators, max_depth=20, class_weight="balanced"
  - `rf_med`: 300 estimators, max_depth=40, class_weight="balanced_subsample"

## Results

### Best Model per Family (Ranked by Validation Accuracy)

| Family | Model Name | Train Acc | Val Acc | Test Acc | Val F1 | Test F1 | Best Epoch |
|--------|-----------|-----------|---------|----------|---------|----------|-------------|
| DNN | dnn_small | ~0.95 | **~0.93** | ~0.92 | ~0.93 | ~0.92 | 8-10 |
| TextCNN | cnn_med | ~0.94 | **~0.92** | ~0.91 | ~0.92 | ~0.91 | 7-9 |
| LSTM | lstm_med | ~0.93 | **~0.91** | ~0.90 | ~0.91 | ~0.90 | 6-8 |
| GRU | gru_large | ~0.92 | **~0.90** | ~0.89 | ~0.90 | ~0.89 | 7-9 |
| SimpleRNN | rnn_small | ~0.96 | **~0.93** | ~0.92 | ~0.93 | ~0.92 | 8-10 |
| NaiveBayes | nb_alpha_0.1 | ~0.86 | **~0.84** | ~0.84 | ~0.84 | ~0.84 | N/A |
| RandomForest | rf_med | ~0.88 | **~0.82** | ~0.81 | ~0.82 | ~0.81 | N/A |

*Note: Exact values may vary slightly. See `/content/models/results/best_models_summary.csv` for precise metrics.*

### Key Observations

1. **Deep Learning Models Outperform Classical ML**: 
   - Best DL model (DNN/SimpleRNN) achieves ~93% validation accuracy
   - Best classical model (Naive Bayes) achieves ~84% validation accuracy
   - ~9% performance gap

2. **SimpleRNN vs LSTM vs GRU**:
   - SimpleRNN and LSTM show similar performance (~91-93%)
   - GRU performs slightly lower but comparable
   - SimpleRNN trains faster but may have gradient issues on longer sequences

3. **TextCNN Performance**:
   - Strong performance (~92% val acc) with medium configuration
   - Efficient architecture for text classification
   - Good balance between accuracy and training time

4. **Classical ML**:
   - Naive Bayes with low smoothing (alpha=0.1) performs best
   - Random Forest shows good performance but slower training
   - Both benefit from TF-IDF feature engineering

5. **Overfitting Analysis**:
   - Most DL models show slight overfitting (train_acc > val_acc by 2-3%)
   - Early stopping effectively prevents severe overfitting
   - Test performance closely matches validation performance

### Training Details

- **Epochs**: Maximum 20 (early stopping typically triggers at 8-12 epochs)
- **Batch Sizes**: 32-64 depending on model size
- **Learning Rates**: 5e-5 to 2e-4 (adaptive reduction)
- **Class Weights**: 
  - Negative: 0.53
  - Neutral: 1.57
  - Positive: 2.07

## Usage

### Running the Notebook

1. **Install Dependencies**:
```python
!pip install -q kagglehub
!pip install -q wordcloud
```

2. **Execute Cells in Order**:
   - Cell 1: Data Loading, Preparation and Preprocessing
   - Cell 2: Class distributions and WordCloud visualizations — Train / Validation / Test
   - Cell 3: Modelling
   - Cell 4: Generate classification reports
   - Cell 5: Create summary tables
   - Cell 6: Generate accuracy plots
   - Cell 7: Generate loss plots
   - Cell 8: Interactive prediction

### Making Predictions

Run the prediction cell (Cell 8) and enter a tweet text:

```python
# Example usage:
Input: "i love u"
Output: Predictions from best model per family:
- DNN: Positive (0.9768)
- TextCNN: Positive (0.8381)
- NaiveBayes: Positive (0.9239)
...
```

### Loading Saved Models

```python
import pickle, joblib
import tensorflow as tf

# Load tokenizer and vectorizer
with open("/content/datasets/keras_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("/content/datasets/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Load Keras model
model = tf.keras.models.load_model("/content/models/dnn_small_best.keras")

# Load Sklearn model
nb_model = joblib.load("/content/models/nb_alpha_0.1_nb.joblib")
```

## Requirements

### Python Packages

```
tensorflow>=2.19.0
numpy
pandas
scikit-learn
matplotlib
seaborn
wordcloud
kagglehub
tqdm
joblib
```

### Key Versions

- **TensorFlow**: 2.19.0
- **Python**: 3.7+ (recommended 3.8+)
- **CUDA**: Optional (for GPU acceleration)

### Hardware Recommendations

- **Minimum**: CPU with 8GB RAM
- **Recommended**: GPU with 4GB+ VRAM (significantly faster training)
- **Training Time**: 
  - CPU: ~2-3 hours for all models
  - GPU: ~30-45 minutes for all models

## Key Findings

### What Worked Well

1. **Text Preprocessing**: Removing mentions and URLs, merging negative reasons improved signal-to-noise ratio
2. **Class Weights**: Effectively handled imbalanced dataset
3. **Early Stopping**: Prevented overfitting while maximizing performance
4. **Ensemble of Architectures**: Different models capture different patterns
5. **TF-IDF + Classical ML**: Naive Bayes with low smoothing performed surprisingly well for a simple model

### Challenges Encountered

1. **Class Imbalance**: Required careful handling with class weights
2. **Overfitting**: Managed through regularization, dropout, and early stopping
3. **Text Variations**: Twitter text with typos, slang, and abbreviations required robust preprocessing
4. **Model Selection**: Multiple strong candidates (DNN, SimpleRNN, TextCNN) made final selection non-trivial

### Future Improvements

1. **Pretrained Embeddings**: Use Word2Vec, GloVe, or BERT embeddings instead of random initialization
2. **Transformer Models**: Implement BERT, RoBERTa, or DistilBERT for potentially higher accuracy
3. **Ensemble Methods**: Combine predictions from multiple best models
4. **Advanced Augmentation**: Apply back-translation or synonym replacement
5. **Feature Engineering**: Incorporate airline metadata, temporal features
6. **Hyperparameter Tuning**: Use Bayesian optimization or Optuna for more systematic search

## Notes

- **Reproducibility**: All random seeds set to 42 for consistent results
- **Memory Management**: Models cleared from memory after evaluation to prevent OOM errors
- **Model Persistence**: All models saved with best validation performance
- **Evaluation**: Metrics computed on held-out test set (not used during training/validation)


## Made By

#### Rohan Karna (M25MAC005)
#### Akshay Gupta (M25MAC017)

---

**Last Updated**: 2025  
**Status**: Complete - All models trained and evaluated

