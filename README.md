# Spam-Detection-ML-Model

## Overview
This project is a **Spam Detection Model** that classifies text messages as **Spam** or **Not Spam** using **Machine Learning**. The model is trained on a dataset of labeled messages and utilizes **TF-IDF vectorization** and a **Logistic Regression classifier**.

**🔗 Try it live:** [Click here](https://spam-detection-model-amyelu.streamlit.app/)

## Features
- Classifies text messages as spam or not spam
- Uses **TF-IDF vectorization** for text preprocessing
- Trained using **Logistic Regression**
- Supports **custom input prediction**

## Installation
To use this model, first clone the repository and install the required dependencies.

```bash
# Clone the repository
git clone https://github.com/yourusername/spam-detection-model.git
cd spam-detection-model

# Install dependencies
pip install -r requirements.txt
```

## Running the Model 
Once dependencies are installed, you can run the model using:

```bash
python main.py
```

This will prompt you to enter a message, and the model will predict whether it is **Spam** or **Not Spam**.

## Saving and Loading the Model
The trained model and vectorizer are saved as `.pkl` files:
- **spam_model.pkl** - The trained ML model
- **vectorizer.pkl** - The TF-IDF vectorizer

If you need to reload them for predictions:
```python
import joblib
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
```

## Example Usage in Python
You can also use the model directly in your Python code:

```python
from main import predict_spam

text = "Congratulations! You have won a free gift."
print(predict_spam(text))  # Output: Spam
```
#### Out-of-Scope Use
This model is not designed to detect phishing attempts, malware in attachments, or other security threats beyond text-based spam classification. Its performance may degrade on texts that differ significantly from the training data, such as non-English messages or content from domains unrelated to emails.

#### Performance & Expected Use Cases
The model achieves high accuracy on standard spam datasets, making it suitable for filtering spam in emails, SMS, and similar text-based communications. However, results may vary depending on the dataset and real-world application, so users should validate performance on their specific data.

#### Biases, Risks, and Limitations
The model's predictions may be influenced by biases in the training data, especially for edge cases or underrepresented categories. While it effectively identifies spam, occasional false positives and negatives may occur. Users should incorporate human oversight or additional review mechanisms to ensure important messages are not mistakenly filtered out.
- **Short texts are more prone to misclassification if they contain biased words that the model has strongly associated with spam.**

#### Model Maintenance & Adaptability
Spam tactics evolve over time, so regularly updating your dataset and retraining the model will help it adapt to new spam patterns and maintain accuracy in real-world applications.

## Contributing
Feel free to fork this repository, create a branch, and submit a pull request with improvements!


