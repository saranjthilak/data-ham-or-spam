# Ham or Spam Email Classification 📧

A machine learning project that classifies emails as spam (1) or normal emails (0) using Natural Language Processing and Multinomial Naive Bayes.

## 🎯 Project Overview

This project implements a complete email spam classification pipeline that:
- Cleans and preprocesses email text data
- Converts text into numerical representations using Bag-of-Words
- Applies Multinomial Naive Bayes for classification
- Achieves **99% accuracy** on the test dataset

## 🚀 Quick Start

### Prerequisites
```bash
pip install nltk pandas scikit-learn
```

### NLTK Setup
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### Data
The project uses the Ham/Spam email dataset from:
```
https://wagon-public-datasets.s3.amazonaws.com/05-Machine-Learning/10-Natural-Language-Processing/ham_spam_emails.csv
```

## 📊 Dataset Information

- **Total emails**: 5,728
- **Features**: Email text content
- **Target**: Binary classification (0 = Ham, 1 = Spam)
- **Text preprocessing**: Punctuation removal, lowercasing, number removal, stopword removal, lemmatization

## 🔧 Text Preprocessing Pipeline

### 1. Punctuation Removal
Removes all punctuation marks from email text using `string.punctuation`.

### 2. Lowercase Conversion
Converts all text to lowercase for consistency.

### 3. Number Removal
Eliminates all numeric characters from the text.

### 4. Stopword Removal
Removes common English stopwords using NLTK's stopwords corpus.

### 5. Lemmatization
Reduces words to their base forms using WordNet Lemmatizer for both verbs and nouns.

## 🤖 Model Architecture

### Bag-of-Words Vectorization
- Uses `CountVectorizer` from scikit-learn
- Creates a sparse matrix representation
- Vocabulary size: 28,173 unique words

### Multinomial Naive Bayes
- Probabilistic classifier ideal for text classification
- Assumes feature independence
- Handles sparse data efficiently

## 📈 Results

### Cross-Validation Performance
```
Cross-validated accuracy scores: [0.9877836, 0.98516579, 0.9921466, 0.98515284, 0.99213974]
Mean accuracy: 0.99 (99%)
```

The model demonstrates excellent performance with consistent accuracy across all folds.

## 🛠️ Usage

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

# Load and preprocess data
df = pd.read_csv("your_email_dataset.csv")
# Apply preprocessing pipeline...

# Vectorize text
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(df['clean_text'])

# Train and evaluate model
clf = MultinomialNB()
accuracy_scores = cross_val_score(clf, X_bow, df['spam'], cv=5, scoring='accuracy')
print(f"Mean accuracy: {accuracy_scores.mean():.2f}")
```

## 📁 Project Structure

```
spam-classification/
├── notebook.ipynb          # Main analysis notebook
├── README.md              # This file
├── requirements.txt       # Dependencies
└── data/
    └── ham_spam_emails.csv # Dataset
```

## 🔍 Key Features

- **Comprehensive text preprocessing** with multiple cleaning steps
- **Robust evaluation** using 5-fold cross-validation
- **High accuracy** (99%) with minimal false positives/negatives
- **Scalable pipeline** that can handle large email datasets
- **Memory efficient** using sparse matrix representations

## 🚀 Future Enhancements

- **Feature Engineering**: TF-IDF vectorization, n-grams
- **Advanced Models**: SVM, Random Forest, Neural Networks
- **Hyperparameter Tuning**: Grid search optimization
- **Deployment**: API endpoint for real-time classification
- **Performance Metrics**: Precision, recall, F1-score analysis

## 📚 Dependencies

- `pandas`: Data manipulation and analysis
- `scikit-learn`: Machine learning algorithms and tools
- `nltk`: Natural language processing toolkit
- `string`: Text processing utilities

## 🎯 Business Applications

This spam classification system can be integrated into:
- Email service providers
- Corporate email security systems
- Marketing automation platforms
- Customer communication filters

## 📄 License

This project is available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for improvements.

---

**Achieved 99% accuracy in email spam classification using NLP and Multinomial Naive Bayes** 🎉
