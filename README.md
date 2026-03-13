# NLP-Based Text Classification: Fake News Detection using LSTM

## Project Overview

This project implements a **Fake News Detection system** using Natural Language Processing (NLP) and a Long Short-Term Memory (LSTM) deep learning model.
The model analyzes news text and classifies whether the news is **real or fake**.

The project demonstrates key NLP and deep learning concepts including:

* Text preprocessing
* Tokenization and sequence padding
* Word embeddings
* LSTM-based classification
* Model training and evaluation

---

## Repository Structure

```
NLP-Based-Text-Classification
│
├── Fake News Detection Analysis - LSTM Classification.ipynb
│
├── train_part1.csv
├── train_part2.csv
├── train_part3.csv
├── train_part4.csv
│
├── test.csv
├── submit.csv
│
└── README.md
```

---

## Dataset

The original **training dataset was larger than GitHub’s file size limit**, so it has been split into four parts:

```
train_part1.csv
train_part2.csv
train_part3.csv
train_part4.csv
```

To run the project, these files must be **merged to recreate the original `train.csv` dataset**.

---

## Recreating the Training Dataset

Run the following Python code to merge the dataset parts:

```python
import pandas as pd

df1 = pd.read_csv("train_part1.csv")
df2 = pd.read_csv("train_part2.csv")
df3 = pd.read_csv("train_part3.csv")
df4 = pd.read_csv("train_part4.csv")

train = pd.concat([df1, df2, df3, df4])
train.to_csv("train.csv", index=False)
```

After running this code, a new file **`train.csv`** will be created.

---

## Requirements

Install the required libraries before running the notebook:

```
pip install numpy pandas matplotlib scikit-learn tensorflow keras nltk
```

---

## How to Run the Project

1. Clone the repository
2. Merge the training dataset parts to create `train.csv`
3. Open the notebook:

```
Fake News Detection Analysis - LSTM Classification.ipynb
```

4. Run all cells sequentially.

---

## Model Used

* Long Short-Term Memory (LSTM)
* Tokenization and padding
* Deep learning text classification

---

## Output

The model predicts whether a news article is:

* **Real**
* **Fake**

Predictions can be generated for the `test.csv` dataset.

---

## Author

**Sumit Joshi**

---

## Notes

The dataset was split due to GitHub file size limitations.
All dataset parts belong to the same original training dataset.
