# Fake News Detection Project

## Overview
This project implements a comprehensive fake news detection system using Natural Language Processing (NLP) techniques. The system uses multiple machine learning and deep learning models to classify news articles as real or fake.

## Industry
**Media**

## Objective
Build a model to classify news articles as real or fake using natural language processing (NLP).

## Features
- **Multiple Model Approaches**: Implements three different models for comparison
  - Logistic Regression with TF-IDF
  - LSTM Neural Network
  - BERT (Bidirectional Encoder Representations from Transformers)
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score
- **False Positive/Negative Analysis**: Detailed confusion matrix analysis
- **Interactive Prediction Interface**: Simple function to predict on new text

## Dataset
- **Source**: Fake News Dataset (Kaggle)
- **Expected Format**: CSV file with columns:
  - `title`: News article title
  - `text`: News article content
  - `label`: Binary label (0=Real, 1=Fake)

### How to Get the Dataset
1. Visit [Kaggle Fake News Dataset](https://www.kaggle.com/datasets)
2. Search for "Fake News" datasets
3. Download the dataset
4. Place it in the same directory as the notebook
5. Update the file path in the notebook (Section 2)

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup Steps

1. **Clone or download this repository**

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (will be done automatically in the notebook):
   - punkt tokenizer
   - stopwords corpus

4. **Download the dataset** from Kaggle and place it in the project directory

## Usage

### Running the Notebook

1. Open `Fake_News_Detection.ipynb` in Jupyter Notebook or JupyterLab

2. **Update the dataset path** in Section 2:
   ```python
   df = pd.read_csv('path/to/your/fake_news_dataset.csv')
   ```

3. Run all cells sequentially

### Project Structure

The notebook is organized into the following sections:

1. **Import Required Libraries**: All necessary imports
2. **Load and Explore Dataset**: Data loading and exploratory analysis
3. **Data Preprocessing**: Text cleaning, tokenization, stemming
4. **Train-Test Split**: 80-20 split with stratification
5. **Model 1 - Logistic Regression**: TF-IDF vectorization + Logistic Regression
6. **Model 2 - LSTM**: Deep learning model with bidirectional LSTM
7. **Model 3 - BERT**: Transformer-based model
8. **Model Comparison**: Side-by-side comparison of all models
9. **Prediction Interface**: Functions to predict on new text
10. **Summary**: Project conclusions and next steps

### Using the Prediction Interface

#### Method 1: Function Call
```python
result = predict_fake_news("Your news article text here", model_type='lr')
print(result)
```

#### Method 2: Interactive Function
```python
interactive_predict()
```

**Available Models:**
- `'lr'`: Logistic Regression (Fastest)
- `'lstm'`: LSTM Neural Network (Moderate speed)
- `'bert'`: BERT (Most accurate, slower)

## Model Performance

The models are evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

Each model includes:
- Classification report
- Confusion matrix visualization
- False positive/negative analysis

## Tools and Technologies

- **Python**: Programming language
- **Pandas & NumPy**: Data manipulation
- **Scikit-learn**: Machine learning (Logistic Regression, TF-IDF)
- **TensorFlow/Keras**: Deep learning (LSTM)
- **Transformers (Hugging Face)**: BERT implementation
- **NLTK**: Natural language processing utilities
- **Matplotlib & Seaborn**: Data visualization

## Project Requirements Checklist

✅ Load text data and preprocess (remove punctuation, tokenize, embed with BERT or GloVe)  
✅ Split into training and test sets  
✅ Train a model (BERT, LSTM, and Logistic Regression) to classify articles  
✅ Evaluate with accuracy, precision, and recall; analyze false positives/negatives  
✅ Create a simple interface to input news text and get predictions  

## Notes

- **BERT Training**: BERT requires significant computational resources. Consider using GPU for faster training.
- **Dataset Size**: The notebook includes sample data structure. Replace with actual dataset for real results.
- **Memory Requirements**: LSTM and BERT models may require substantial RAM (8GB+ recommended).

## Future Enhancements

- Ensemble methods combining multiple models
- Additional features (author, source, publication date)
- Web application deployment
- Real-time news verification API
- Multi-language support

## License
This project is provided for educational purposes as part of the self learning program.

## Author
Priyanka Gowda

## Last Updated
January 2025

