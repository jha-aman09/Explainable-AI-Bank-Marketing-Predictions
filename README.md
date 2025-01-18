# 📊 Bank Marketing Predictions

This repository demonstrates the use of **Random Forest Classifier**, **LIME**, and **SHAP** for predicting customer response to marketing campaigns using a dataset related to bank marketing. The model's results are explained using explainable AI techniques, enabling better interpretability of predictions.

---

## 🗂️ Files Included

- `bank-additional-full.csv`: The dataset used for training and testing the machine learning model. It contains demographic and marketing campaign data.
- `Bank_Marketing_Predictions.py`: Python script implementing data preprocessing, model training, evaluation, and explainability using LIME and SHAP.

## 🚀 Google Colab

You can run the code interactively on Google Colab by accessing the notebook [here](https://colab.research.google.com/drive/1N6BhyB0wNIQ_6Oit0RFPnTs6hTLFB8AN?usp=sharing).

## 👨‍💻 Author

This project was created by **Aman Jha**. Feel free to reach out for any questions or suggestions!  
- GitHub: [Aman Jha](https://github.com/jha-aman09)  
- LinkedIn: [Aman Jha](https://www.linkedin.com/in/aman--jha)

## 📈 Key Features

- **Random Forest Classifier** for predictive modeling
- **LIME (Local Interpretable Model-agnostic Explanations)** for explaining individual predictions
- **SHAP (SHapley Additive exPlanations)** for global feature importance analysis
- **Data Visualization** using matplotlib for feature importance insights

## 📝 Steps to Run the Bank Marketing Predictions Script

Follow the steps below to run the `Bank_Marketing_Predictions.py` file on your local machine or in a Google Colab environment:

### 1. **Clone the Repository**

First, clone the repository to your local machine or download the files manually.

To clone via Git, use the following command in your terminal or command prompt:

```bash
git clone https://github.com/jha-aman09/Explainable-AI-Bank-Marketing-Predictions.git
```

### 2. **Install Required Libraries**

You need to install the necessary Python libraries to run the script. These can be installed using `pip`:

```bash
pip install -r requirements.txt
```

If the `requirements.txt` file is not available, manually install the necessary packages by running:

```bash
pip install pandas numpy scikit-learn lime shap matplotlib
```

### 3. **Prepare Your Environment**

Ensure that you have Python (preferably version 3.7 or higher) installed on your machine.

### 4. **Download the Dataset**

Download the `bank-additional-full.csv` file.

### 5. **Run the Script**

Once you have the necessary libraries installed and dataset in place, run the Python script using:

```bash
python Bank_Marketing_Predictions.py
```

### 6. **View the Results**

After running the script, the following outputs will be displayed:
- **Model Accuracy**: Shows the classification accuracy of the Random Forest model.
- **Classification Report**: Displays precision, recall, and F1-score metrics for each class.
- **Feature Importance Plot**: A bar plot showing the importance of each feature in the model's decision-making process.
- **LIME Explanations**: Visual explanations for individual predictions will be shown using LIME's `explain_instance` method, along with corresponding plots.

### 7. **Optional: Use Google Colab**

If you prefer running the script on Google Colab, follow these steps:
1. Open the [Google Colab notebook](https://colab.research.google.com/drive/1N6BhyB0wNIQ_6Oit0RFPnTs6hTLFB8AN).
2. Upload the dataset `bank-additional-full.csv` into the Colab environment.
3. Run the code cells sequentially to execute the predictions and visualizations.
