# Zomato-Review-Segmentation-Using-ML
# Customer Review Clustering using NLP and Machine Learning

## 📌 Project Overview

This project focuses on analyzing and clustering customer reviews using Natural Language Processing (NLP) and Machine Learning techniques. The objective is to automatically group similar customer reviews into meaningful clusters to help businesses understand customer behavior, preferences, and feedback.

The project applies text preprocessing, feature engineering, dimensionality reduction, and clustering algorithms to identify hidden patterns in customer reviews.

---

## 🎯 Problem Statement

Businesses receive thousands of customer reviews through online platforms. Manually analyzing these reviews is time-consuming and inefficient. This project aims to automatically segment customer reviews into meaningful groups using machine learning clustering techniques to support better business decisions and customer understanding.

---

## 🚀 Features

* Text preprocessing and cleaning
* TF-IDF vectorization
* Feature engineering
* Dimensionality reduction using TruncatedSVD
* K-Means Clustering
* Agglomerative Clustering
* DBSCAN Clustering
* Hyperparameter tuning
* Model comparison using Silhouette Score
* Model deployment using Joblib
* Prediction on unseen customer reviews

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* Scikit-learn
* Matplotlib
* Seaborn
* SciPy
* Joblib
* Google Colab

---

## 📂 Project Workflow

1. Data Collection
2. Data Cleaning
3. Text Preprocessing
4. TF-IDF Vectorization
5. Feature Engineering
6. Data Transformation
7. Feature Scaling
8. Dimensionality Reduction (TruncatedSVD)
9. Clustering Model Development
10. Hyperparameter Tuning
11. Model Evaluation
12. Model Saving
13. Deployment Sanity Check

---

## 📊 Models Implemented

| Model                    | Silhouette Score |
| ------------------------ | ---------------- |
| K-Means                  | 0.3688           |
| Agglomerative Clustering | 0.3151           |
| DBSCAN                   | 0.0785           |

---

## 🏆 Best Performing Model

**K-Means Clustering** achieved the highest Silhouette Score and was selected as the final model.

* Number of clusters: 5
* Silhouette Score: 0.3688

---

## 📈 Evaluation Metric

The models were evaluated using the **Silhouette Score**, which measures:

* Cluster cohesion
* Cluster separation

Higher values indicate better clustering performance.

---

## 🔧 Hyperparameter Tuning

### K-Means

* Number of clusters (k)

### Agglomerative Clustering

* Linkage methods

### DBSCAN

* Epsilon (eps)

---

## 💾 Saved Models

* kmeans_model.pkl
* tfidf_vectorizer.pkl
* svd_model.pkl
* scaler.pkl

---

## 🧪 Deployment Sanity Check

The saved model was successfully loaded and tested on unseen customer reviews. The model correctly predicted the customer review cluster, confirming deployment readiness.

---

## 📁 Project Structure

```text
Customer-Review-Clustering/
│
├── data/
├── notebooks/
├── models/
│   ├── kmeans_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── svd_model.pkl
│   └── scaler.pkl
│
├── images/
├── README.md
└── requirements.txt
```

---

## ▶️ Installation

```bash
git clone https://github.com/yourusername/Customer-Review-Clustering.git

cd Customer-Review-Clustering

pip install -r requirements.txt
```

---

## 🚀 Run the Project

Open the notebook in Google Colab or Jupyter Notebook and execute all cells.

---

## 📌 Business Impact

* Customer segmentation
* Review analysis
* Customer behavior understanding
* Service improvement
* Better business decision-making
* Targeted marketing strategies

---

## 👨‍💻 Author

Rudra Pratap Singh

---

## ⭐ If you found this project useful, please consider giving it a star.
