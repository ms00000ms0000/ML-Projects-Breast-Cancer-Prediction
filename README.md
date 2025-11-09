# 🎗️ ML Project - Breast Cancer Prediction  

## 🚀 Overview  
This **Machine Learning project** focuses on **Breast Cancer Classification** using the **Logistic Regression** algorithm.  
The model utilizes the **predefined Breast Cancer dataset from Scikit-learn**, which contains key diagnostic features used to classify tumors as **Benign (1)** or **Malignant (0)**.  

The dataset was processed and trained using supervised learning techniques to accurately distinguish between the two classes.  
The model achieved an impressive **accuracy of 92%**, showcasing its reliability in early cancer detection.  

This project highlights the power of **Machine Learning in healthcare**, aiding in **medical diagnosis and decision support** to improve patient outcomes through data-driven insights.  

---

## 🔍 About the Project  
The **Breast Cancer Prediction System** leverages logistic regression to perform binary classification on medical data.  
By learning from the characteristics of cell nuclei present in breast mass samples, the model can accurately determine whether the tumor is **malignant (cancerous)** or **benign (non-cancerous)**.  

This project demonstrates how **ML algorithms** can assist medical professionals by providing **quick, data-backed diagnostic insights** that enhance decision-making and potentially save lives.  

---

## 🧠 Model Architecture  
The project pipeline follows these steps:  

1. **Dataset Loading** – Using the built-in **Breast Cancer dataset** from Scikit-learn.  
2. **Data Preprocessing** – Converting the dataset into a Pandas DataFrame for exploration and cleaning.  
3. **Feature Selection** – Identifying the most relevant features for classification.  
4. **Data Splitting** – Dividing the dataset into training and testing sets using **train_test_split**.  
5. **Model Training** – Training a **Logistic Regression** model on the training data.  
6. **Evaluation** – Measuring accuracy, confusion matrix, and classification report.  

---

## 🧾 Dataset Description  
The dataset is provided by **Scikit-learn** and contains **569 instances** with **30 numerical features**, each representing a property of a cell nucleus.  

| Feature | Description |
|----------|-------------|
| **mean radius** | Average size of cell nuclei |
| **mean texture** | Variation in cell texture |
| **mean perimeter** | Mean perimeter of cell nuclei |
| **mean area** | Average cell area |
| **mean smoothness** | Consistency of cell surface |
| **...** | 30 total diagnostic attributes |
| **target** | 0 = Malignant, 1 = Benign |

---

## ⚙️ Tech Stack & Libraries  

**Language:**  
* Python 🐍  

**Libraries:**  
* **NumPy** – Numerical computation  
* **Pandas** – Data manipulation  
* **Matplotlib / Seaborn** – Data visualization  
* **Scikit-learn** – Model building, training, and evaluation  

---

## 🚀 Features  
* Predicts whether a tumor is **benign or malignant**  
* Uses **Scikit-learn’s Breast Cancer dataset**  
* Implements **Logistic Regression** for classification  
* Performs **EDA** to understand feature correlations  
* Provides **accuracy score**, **confusion matrix**, and **classification report**  
* Demonstrates **ML in healthcare** applications  

---

## 📊 Results  
The **Logistic Regression model** achieved:  
* **Accuracy:** 92%  
* **High Precision and Recall** in identifying both benign and malignant cases  

This confirms the model’s **effectiveness and dependability** in medical diagnosis.  

---

## 📁 Repository Structure  

```
📦 ML-Projects-Breast-Cancer-Prediction
│
├── Breast_Cancer_Prediction.ipynb # Main Jupyter Notebook
├── requirements.txt # Dependencies and libraries
└── README.md # Project documentation
```

---

## 🧪 How to Run  

1. **Clone the repository:**  
   ```bash
   git clone https://github.com/ms00000ms0000/ML-Projects-Breast-Cancer-Prediction.git
   cd ML-Projects-Breast-Cancer-Prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook:**
   ```bash
   jupyter notebook Breast_Cancer_Prediction.ipynb
   ```

4. **View the results:**
* The notebook displays EDA visuals, model training outputs, and prediction results.

---

## 📈 Future Improvements

* Integrate Deep Learning models for higher accuracy

* Develop a web-based prediction interface using Streamlit or Flask

* Enhance the model with feature scaling and cross-validation

* Explore ensemble methods like Random Forest or Gradient Boosting

---

## 👨‍💻 Developer

Developed by: Mayank Srivastava
