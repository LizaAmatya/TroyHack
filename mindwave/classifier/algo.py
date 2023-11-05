from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

# Load Text Cleaning Pkgs
import neattext.functions as nfx
# Load ML Pkgs
# Estimators
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
# Transformers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def classifier():
    data = pd.read_csv(
        "/mnt/A69EB3A19EB36907/Troy Uni Study/TroyHack/mindwave/classifier/detection_updated.csv", encoding="ISO-8859-1", usecols=['text','class'])

    data['class'] = data['class'].replace(np.nan, 'unknown')
    # print('class data------------------', data)

    # print(data.head())
    data.head()
    
    
    # # User handles
    data['clean_text'] = data['text'].apply(
        lambda x: nfx.remove_userhandles(x) if isinstance(x, str) else x)
    # Clean Stopwords
    data['clean_text'] = data['text'].apply(
        lambda x: nfx.remove_stopwords(x) if isinstance(x, str) else x)

    # print(data['clean_text'])
    # Encode Labels
    # label_encoder = LabelEncoder()
    # data['class'] = label_encoder.fit_transform(data['class'].astype(str))
    print(data)

    Xfeatures = data['clean_text']
    ylabels = data['class']

    x_train, x_test, y_train, y_test = train_test_split(
        Xfeatures, ylabels, test_size=0.3, random_state=42)

    x_train = x_train.apply(lambda x: str(x).lower())
    x_test = x_test.apply(lambda x: str(x).lower())

    # LogisticRegression Pipeline
    pipe_lr = Pipeline(steps=[('cv', CountVectorizer()),
                            ('lr', LogisticRegression())])
    
    # Train and Fit Data
    pipe_lr.fit(x_train, y_train)

    # Check Accuracy
    score = pipe_lr.score(x_test, y_test)
    
    print(score)
    return pipe_lr

# Make A Prediction
pipe_lr = classifier()
predict_data = "I'm just feeling really down today. I can't seem to shake off this feeling."
print('result', pipe_lr.predict([predict_data]))
