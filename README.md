# 🧠 Breast Cancer Classification (AI in 15)

A simple machine learning project that trains a neural network to classify breast cancer tumors as **malignant (1)** or **benign (0)** using the [Breast Cancer Wisconsin Dataset](https://gist.github.com/KhanradCoder/35a6beea49e5b9ba62797e595a9626c0).

This project was originally adapted from a Colab notebook titled **AI in 15**, and demonstrates a full deep learning workflow in just a few lines of Python.

---

## 🚀 Features
- Loads and preprocesses a breast cancer dataset.
- Splits data into training and testing sets.
- Builds a TensorFlow neural network for binary classification.
- Trains and evaluates the model with progress logs.
- Prints clean, human-readable messages during execution.

---

## 🧰 Requirements

Make sure you have Python 3.9+ installed.

### Using [uv](https://github.com/astral-sh/uv) (recommended):
```bash
# Create and activate environment
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
uv pip install pandas scikit-learn tensorflow matplotlib
