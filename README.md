# SCREENSHOT
<img width="1139" height="649" alt="Screenshot 2025-11-17 000435" src="https://github.com/user-attachments/assets/80709ee3-1e45-4617-90f4-6ec29d15c521" />


# 📱 SpamShield – SMS Spam Detection Model

A machine-learning powered SMS classification system that detects whether a given message is **Spam** or **Not Spam**.
This project uses Natural Language Processing (NLP) and supervised learning to classify text messages with high accuracy.

## 🚀 Features

* ✔️ Cleaned & preprocessed text (lowercasing, stopwords removal, stemming, punctuation removal)
* ✔️ Feature extraction using **Bag of Words / TF-IDF**
* ✔️ Model training using **Naive Bayes / Logistic Regression / SVM**
* ✔️ High accuracy on SMS Spam Classification dataset
* ✔️ Interactive Web App (Streamlit / Flask)
* ✔️ Easy to integrate with any backend
```
## 📂 Project Structure
SpamShield/
│── data/
│   └── spam.csv
│── notebooks/
│   └── EDA_and_Model.ipynb
│── models/
│   └── spam_classifier.pkl
│   └── vectorizer.pkl
│── app/
│   └── app.py
│── README.md
│── requirements.txt
│── utils.py
│── preprocessing.py
```
## 🧠 How It Works

1. **Data Collection**
   The project uses a labeled SMS dataset containing two categories:

   * `ham` → Not Spam
   * `spam` → Unwanted promotional/fraud message

2. **Data Cleaning**

   * Lowercasing
   * Stopwords removal
   * Stemming
   * Removing special characters, numbers, punctuation

3. **Feature Engineering**
   Text is converted into numerical features using:

   * **CountVectorizer (Bag of Words)**
   * **TF-IDF Vectorizer**

4. **Model Training**
   Different machine learning models are trained and evaluated:

   * Naive Bayes
   * Logistic Regression
   * SVM
   * Random Forest (optional)

   Best performing model is saved as:
   ```
   spam_classifier.pkl
   vectorizer.pkl
   ```

5. **Deployment**
   A lightweight web UI / API (Flask or Streamlit) is created for real-time prediction.

---

## 🛠️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/SpamShield.git
cd SpamShield
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit/Flask App

If using **Streamlit**:

```bash
streamlit run app/app.py
```
## 🔍 Usage

### Example input:

```
Congratulations! You won a free mobile recharge. Click here to claim.
```

### Model output:

```
SPAM
```

---

## 📈 Model Performance (sample)

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 97%   |
| Precision | 96%   |
| Recall    | 95%   |
| F1-Score  | 95%   |

---

## 🧪 Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* NLTK
* Matplotlib / Seaborn (for EDA)
* Streamlit / Flask (Deployment)

---

## 📦 Exported Files

| File                  | Description                         |
| --------------------- | ----------------------------------- |
| `spam_classifier.pkl` | Trained classification model        |
| `vectorizer.pkl`      | CountVectorizer / TF-IDF vectorizer |
| `app.py`              | Web interface for predictions       |
| `requirements.txt`    | Project dependencies                |

---

## 🤖 Future Improvements

* Deep learning model (LSTM / BERT)
* Multilingual spam detection
* API for WhatsApp / SMS gateways
* Dashboard for monitoring spam patterns

---
