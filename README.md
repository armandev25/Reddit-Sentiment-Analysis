# Reddit-Sentiment-Analysis

## Project Overview

This project performs sentiment analysis on Reddit comments using two approaches:  
- **VADER**: A lexicon and rule-based sentiment analysis tool specially tuned for social media text.  
- **Logistic Regression**: A classic machine learning classifier trained on TF-IDF features extracted from cleaned text.

The goal is to compare the effectiveness of a lexicon-based method versus a supervised ML model on this dataset, analyze errors, and extract meaningful insights.

---

## Dataset

The dataset is sourced from [Kaggle - Twitter and Reddit Sentimental Analysis Dataset](https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset).

- **Dataset Shape**: (37249, 2)  
- **Sentiment Distribution**:  
  | Sentiment Label | Count  | Description          |  
  |-----------------|--------|----------------------|  
  | 1               | 15771  | Positive sentiments   |  
  | 0               | 12778  | Neutral sentiments    |  
  | -1              | 8250   | Negative sentiments   |

I created a cleaned version of this dataset (`cleaned_dataset.csv`) after text preprocessing steps including noise removal, tokenization, and normalization.

---

## VADER Sentiment Analysis

VADER is effective for social media text due to its ability to handle slang, emoticons, and capitalization nuances. It assigns sentiment scores and labels each comment accordingly.

---

## Why Logistic Regression?

I chose Logistic Regression because:  
- It is simple, interpretable, and fast to train.  
- Well-suited for high-dimensional sparse data like TF-IDF vectors.  
- Provides meaningful coefficients useful for feature importance.  
- Acts as a strong baseline before exploring more complex models.

---

## Insights

- **VADER** performed well on clearly positive/negative comments but struggled more with neutral or ambiguous sentiments.  
- **Logistic Regression** achieved better overall accuracy and provided insights into influential words driving sentiment classification.  
- Error analysis revealed common misclassification patterns such as sarcastic or context-dependent comments.

---

## Model Comparison & Visualization

| Model              | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| VADER              | 0.644284      | 0.662202       | 0.644284    | 0.648424      |
| Logistic Regression | 0.837500      | 0.838995       | 0.837500    | 0.832789      |

### Sample Visualizations

![Confusion Matrix](<img width="568" height="453" alt="image" src="https://github.com/user-attachments/assets/39e3575e-fd30-4f32-9a4b-e266bd5b0f37" />
)  
*Confusion matrix comparing true vs predicted labels for Logistic Regression*

![Feature Importance Word Cloud](<img width="794" height="427" alt="image" src="https://github.com/user-attachments/assets/d80e6bc0-9d3d-43c6-8a20-be59ec15bc79" />
)  
*Word cloud showing top positive and negative words from Logistic Regression coefficients*

![Top Bigrams Bar Chart](<img width="937" height="545" alt="image" src="https://github.com/user-attachments/assets/118f5011-adde-4a6d-804a-9c844bce2f7d" />
)  
*Bar chart displaying the frequency of the top bigrams in the dataset*

---

## Dependencies

- Python 3.x  
- pandas  
- scikit-learn  
- matplotlib  
- seaborn  
- wordcloud  
- nltk (for VADER)

---

## Future Work

- Extend to deep learning models like LSTM or BERT  
- Analyze sentiment over time or by subreddit categories  
- Build interactive dashboard for live sentiment tracking

---

## Contact

Your Name — [Arman Singh](https://www.linkedin.com/in/arman-singh-9bb83628a/)
Mail  — armansr205@gmail.com
