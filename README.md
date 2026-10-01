## Spam Mail Detector

A machine learning project that classifies messages as **Spam** or **Not Spam (Ham)** using Natural Language Processing (NLP). The project also includes a dark-themed **Streamlit web application** where users can enter a message and receive an instant prediction.

##  Features

* Complete ML pipeline: data loading → text preprocessing → TF-IDF → train/test split → model training → evaluation
* Compares **Naive Bayes** and **Logistic Regression**
* Predicts whether a message is **SPAM** or **NOT SPAM**
* Displays prediction confidence and spam probability
* Shows words that contribute to the spam prediction using Logistic Regression coefficients
* Allows users to switch between the two trained models
* Interactive **Streamlit web application**


##  Tech Stack

**Python · Pandas · NLTK · Scikit-learn · Matplotlib · Seaborn · Streamlit · Joblib**

##  Project Workflow

## 1. Data Loading

Loaded the SMS Spam Collection dataset and converted the labels into numerical form:

* Ham = `0`
* Spam = `1`

## 2. Text Preprocessing

The message text was cleaned using NLP techniques:

* Converted text to lowercase
* Removed punctuation and digits
* Tokenized the text
* Removed stopwords using NLTK

## 3. Train-Test Split

The dataset was divided into:

* **80% Training Data**
* **20% Testing Data**

A **stratified split** was used to preserve the class distribution.

## 4. Feature Extraction

The cleaned text was converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

The model used:

* Unigrams and bigrams
* Maximum of 5,000 features
* TF-IDF fitted only on the training data

## 5. Model Training

Two classification algorithms were trained and compared:

* **Multinomial Naive Bayes**
* **Logistic Regression**

## 6. Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

## 7. Deployment

The trained models were integrated into a **Streamlit web application** for real-time spam detection.

##  Results

The models produced the following results on the test dataset:

| Model                   |   Accuracy | Spam Precision | Spam Recall | Spam F1-Score |
| ----------------------- | ---------: | -------------: | ----------: | ------------: |
| **Naive Bayes**         | **96.94%** |    **100.00%** |  **77.18%** |    **87.12%** |
| **Logistic Regression** | **97.48%** |     **90.07%** |  **91.28%** |    **90.67%** |



**Naive Bayes**

* 963 ham messages correctly classified
* 115 spam messages correctly classified
* 0 ham messages incorrectly classified as spam
* 34 spam messages incorrectly classified as ham

**Logistic Regression**

* 948 ham messages correctly classified
* 136 spam messages correctly classified
* 15 ham messages incorrectly classified as spam
* 13 spam messages incorrectly classified as ham

Because the dataset is imbalanced, **accuracy alone does not provide the complete picture**. Precision, recall and F1-score for the spam class provide additional information about false positives and false negatives.


##  How to Run

## 1. Clone the repository

```bash
git clone https://github.com/<your-username>/spam-mail-detector.git
cd spam-mail-detector
```

## 2. Install the required libraries

```bash
pip install -r requirements.txt
```

## 3. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application loads the trained models and TF-IDF vectorizer from the `models/` folder.

To retrain the models, run the cells in `Code.ipynb`.

##  Project Structure

```text
Spam_Mail_Detector/
│
├── Code.ipynb
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── naive_bayes_model.pkl
│   └── logistic_regression_model.pkl
│
├── results/
│   ├── model_comparison.csv
│   └── confusion_matrix.png
│
└── screenshots/
    └── streamlit_app.png
```

##  Limitations

* The model is trained on **SMS messages**, so performance may differ on real email data.
* It primarily uses message text and does not analyze sender information, URLs, attachments or email headers.
* The current model is designed for **English text**.

##  Future Improvements

* Train and evaluate the system using a real-world **email dataset such as the Enron Email Dataset**
* Perform hyperparameter tuning using **GridSearchCV**
* Experiment with additional algorithms such as SVM and Random Forest
* Explore word embeddings and advanced NLP techniques
* Deploy the application online using **Streamlit Community Cloud**

