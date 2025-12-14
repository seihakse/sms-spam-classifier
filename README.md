# email-spam-classifier-new
End to end code for the email spam classifier project

# SMS / Email Spam Classifier

## 1. Project Overview

This project is a **machine learning–based SMS and Email Spam Classifier** built using **Python, Scikit-learn, NLTK, and Streamlit**. The application allows users to input a text message and receive a prediction indicating whether the message is **Spam** or **Ham (Not Spam)**.

The project was originally cloned from an open-source repository and later **refined, fixed, retrained, and deployed** to ensure correctness, compatibility, and cloud deployment stability.

---

## 2. Objectives

* Detect spam messages using Natural Language Processing (NLP)
* Apply text preprocessing and feature extraction using TF-IDF
* Train and deploy a machine learning classification model
* Provide a simple and interactive web interface
* Ensure the project is reproducible and cloud-deployable

---

## 3. Technology Stack

### Programming Language

* Python 3

### Libraries & Tools

* **Streamlit** – Web application framework
* **Scikit-learn** – Machine learning algorithms
* **NLTK** – Text preprocessing and NLP utilities
* **Pandas / NumPy** – Data handling
* **Pickle** – Model serialization
* **Git & GitHub** – Version control
* **Streamlit Community Cloud** – Free deployment platform

---

## 4. Dataset

The model is trained on a public **SMS Spam dataset**, where:

* `spam` → unwanted or promotional messages
* `ham` → legitimate messages

Each message is labeled and used to train a supervised classification model.

---

## 5. Text Preprocessing Pipeline

To ensure consistency and accuracy, the same preprocessing steps are applied during **training and inference**.

### Steps:

1. Convert text to lowercase
2. Tokenize text using `wordpunct_tokenize` (cloud-safe)
3. Remove non-alphanumeric tokens
4. Remove English stopwords
5. Apply stemming using Porter Stemmer

### Final Output

A cleaned and normalized text string suitable for vectorization.

---

## 6. Feature Extraction (TF-IDF)

### Technique Used

* **TF-IDF (Term Frequency – Inverse Document Frequency)**

### Purpose

* Convert text data into numerical vectors
* Assign higher importance to informative words

### Important Note

The TF-IDF vectorizer is **fitted during training** and then **saved** using pickle. The same fitted vectorizer is reused during prediction to maintain feature consistency.

---

## 7. Machine Learning Model

### Algorithm

* **Multinomial Naive Bayes**

### Reason for Choice

* Performs well on text classification tasks
* Efficient and fast
* Suitable for TF-IDF features

### Output

* `1` → Spam
* `0` → Ham (Not Spam)

---

## 8. Model Training & Serialization

During training:

* Text is preprocessed
* TF-IDF vectorizer is fitted
* Model is trained on vectorized text
* Both model and vectorizer are saved as:

```
model.pkl
vectorizer.pkl
```

These files are later loaded directly in the Streamlit application.

---

## 9. Streamlit Application Architecture

### Main Components

* **Input Area** – User enters SMS or email text
* **Preprocessing Module** – Cleans input text
* **Vectorization Module** – Converts text to TF-IDF vectors
* **Prediction Module** – Uses trained model
* **Result Display** – Shows Spam or Ham result

### Key Design Decisions

* Use of `@st.cache_resource` for performance
* Cloud-safe tokenizer to avoid deployment errors
* Input validation to prevent empty predictions

---

## 10. Deployment

### Platform

* **Streamlit Community Cloud (Free)**

### Deployment Steps

1. Push project to GitHub
2. Add `requirements.txt`
3. Configure Streamlit app entry point (`app.py`)
4. Deploy using Streamlit Cloud dashboard

### Deployment Challenges Solved

* Fixed Scikit-learn pickle incompatibility
* Removed Punkt tokenizer dependency
* Ensured NLTK resources download correctly in cloud

---

## 11. Project Structure

```
sms-spam-classifier/
│
├── app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
└── README.md
```

---

## 12. Learning Outcomes

* Understanding NLP preprocessing pipelines
* Proper use of TF-IDF and Naive Bayes
* Handling model serialization issues
* Debugging cloned ML projects
* Cloud deployment of ML applications
* Version control best practices

---

## 13. Conclusion

This project demonstrates a complete **end-to-end machine learning workflow**, from data preprocessing and model training to deployment and user interaction. The refinements made ensure the application is **robust, reproducible, and production-ready** for academic and demonstration purposes.

---

## 14. Future Improvements

* Display prediction confidence scores
* Support batch message classification
* Improve UI/UX design
* Add model evaluation metrics
* Experiment with advanced classifiers (SVM, Logistic Regression)

---

## 15. References

* Scikit-learn Documentation
* NLTK Documentation
* Streamlit Documentation
* Public SMS Spam Dataset

---

**Author:** Seihak
**Project Type:** Academic / Learning Project
**Deployment:** Streamlit Community Cloud
