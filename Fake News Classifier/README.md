
# Detecting Fake News with an ML Classifier

## Project Details

What is Fake News? 

"Fake news, also known as junk news, pseudo-news, alternative facts or hoax news, is a form of news consisting of deliberate disinformation or hoaxes spread via traditional news media or online social media. Digital news has brought back and increased the usage of fake news, or yellow journalism." (credits: Wikipedia)

In this project, we will be detecting Fake political news.

### Technical Terms

TF: Term Frequency -> The number of times a word appears in a document is its Term Frequency.
IDF: Inverse Document Frequency -> Words that occur many times a document, but also occur many times in many others, may be irrelevant. IDF is a measure of how significant a term is in the entire corpus.

The TfidfVectorizer converts a collection of raw documents into a matrix of TF-IDF features.

Goal: To build a model to accurately classify a piece of news as REAL or FAKE
We build a TfidfVectorizer on our dataset. Then, we initialize a PassiveAggressive Classifier and fit the model. In the end, the accuracy score and the confusion matrix tell us how well our model fares.

A PassiveAgressive Classifier is one which remains passive for a correct classification outcome, and turns aggressive in the event of a miscalculation, updating and adjusting. Its purpose is to make updates that correct the net loss. 

## Fake News Detection

Dataset Details: The first column identifies the news, the second and third are the title and text, and the fourth column has labels denoting whether the news is REAL or FAKE.


```python
# Imports
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
```


```python
df = pd.read_csv("news/news.csv")
df.shape
df.info()
df.head()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6335 entries, 0 to 6334
    Data columns (total 4 columns):
    Unnamed: 0    6335 non-null int64
    title         6335 non-null object
    text          6335 non-null object
    label         6335 non-null object
    dtypes: int64(1), object(3)
    memory usage: 198.0+ KB
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>title</th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8476</td>
      <td>You Can Smell Hillary’s Fear</td>
      <td>Daniel Greenfield, a Shillman Journalism Fello...</td>
      <td>FAKE</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10294</td>
      <td>Watch The Exact Moment Paul Ryan Committed Pol...</td>
      <td>Google Pinterest Digg Linkedin Reddit Stumbleu...</td>
      <td>FAKE</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3608</td>
      <td>Kerry to go to Paris in gesture of sympathy</td>
      <td>U.S. Secretary of State John F. Kerry said Mon...</td>
      <td>REAL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10142</td>
      <td>Bernie supporters on Twitter erupt in anger ag...</td>
      <td>— Kaydee King (@KaydeeKing) November 9, 2016 T...</td>
      <td>FAKE</td>
    </tr>
    <tr>
      <th>4</th>
      <td>875</td>
      <td>The Battle of New York: Why This Primary Matters</td>
      <td>It's primary day in New York and front-runners...</td>
      <td>REAL</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Observe Labels
news_labels = df.label
news_labels.head()
```




    0    FAKE
    1    FAKE
    2    REAL
    3    FAKE
    4    REAL
    Name: label, dtype: object




```python
# Split data into train and test data
x_train,x_test,y_train,y_test=train_test_split(df['text'], news_labels, test_size=0.2, random_state=7)
```

Now, let's initialze the TfIDFVectorizer with stop words from the English language and a maximum document frequency of 0.7 (terms with a higher document frequency will be discarded). Stop words are the most common words in a language that are to be filtered out before processing the natural language data. 


```python
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
# Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)
```

Next, we’ll initialize a PassiveAggressiveClassifier. We’ll fit this on tfidf_train and y_train.
Then, we’ll predict on the test set from the TfidfVectorizer and calculate the accuracy with an accuracy score


```python
# Initialize a PassiveAggressiveClassifier
pa_c=PassiveAggressiveClassifier(max_iter=50)
pa_c.fit(tfidf_train,y_train)
# Predict on the test set and calculate accuracy
y_pred=pa_c.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')
```

    Accuracy: 92.82%
    

We got an accuracy of 92.82% with this model, which is pretty good! Finally, let’s print out a confusion matrix to gain insight into the number of false and true negatives and positives.


```python
#confusion_matrix(y_test, y_pred)
confusion_matrix(y_test, y_pred, labels = ['FAKE', 'REAL'])
```




    array([[590,  48],
           [ 43, 586]], dtype=int64)



We have 590 True positives, 586 True negatives, 48 False positives and 43 False negatives. We can see that about 590 news articles were identified to be FAKE News and 586 to be REAL news.


```python
# Misclassification rate
mcl_r = ((48+43)*100)/(590+48+43+586)
print("The misclassification rate of our model was found to be: " + str(round(mcl_r,4)) + "%")
```

    The misclassification rate of our model was found to be: 7.1823%
    


```python

```
