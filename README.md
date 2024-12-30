# NLP-FakeNewsClassifierUsingLSTM
# Fake News Classifier using LSTM 

## Overview
This project implements a deep learning-based fake news classification model using Long Short-Term Memory (LSTM) neural networks. The notebook demonstrates text classification on a fake news dataset, covering preprocessing techniques and model training.

## Dataset
- **Source**: Kaggle Fake News Classification Competition
- **Columns**: 
  - `id`
  - `title`
  - `author`
  - `text`
  - `label`
- **Binary classification**:
  - `1`: Fake News
  - `0`: Real News

## Key Components
### Data Preprocessing
- Cleaning text data
- Tokenization
- One-hot encoding
- Padding sequences

### Model Architecture
- **Embedding Layer**: 40 features
- **Dropout Layers**: To prevent overfitting
- **LSTM Layer**: 100 units
- **Sigmoid Output Layer**: For binary classification

## Performance Metrics
- **Accuracy**: 90.37%
- **Precision**: 
  - Fake News (Class 1): 0.89
  - Real News (Class 0): 0.91
- **Recall**: 
  - Fake News (Class 1): 0.89
  - Real News (Class 0): 0.92

## Requirements
- Python 3.x
- TensorFlow
- Keras
- Pandas
- NumPy
- scikit-learn
- NLTK

## Key Libraries Used
- **Preprocessing**: NLTK, Keras Tokenizer
- **Model**: TensorFlow, Keras
- **Evaluation**: scikit-learn

## How to Use
1. Install the required libraries:
   ```bash
   pip install tensorflow keras pandas numpy scikit-learn nltk
   ```
2. Download the dataset from the [Kaggle Fake News Classification Competition](https://www.kaggle.com).
3. Run the notebook step by step.
4. Train the model on the dataset.
5. Evaluate the performance using metrics like accuracy, precision, and recall.

## Potential Improvements
- Experiment with different embedding dimensions.
- Try a bidirectional LSTM for improved performance.
- Implement more advanced text preprocessing techniques.
- Use pre-trained word embeddings like GloVe or Word2Vec.

## License
This project is open-source and intended for educational purposes.

## Credits
- Dataset from the [Kaggle Fake News Classification Competition](https://www.kaggle.com).
