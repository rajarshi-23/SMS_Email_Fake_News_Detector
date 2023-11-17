import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

sms_tfidf = pickle.load(open('sms_vectorizer.pkl', 'rb'))
sms_model = pickle.load(open('sms_model.pkl', 'rb'))

st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter the sms")

if st.button('Predict SMS'):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input_sms = sms_tfidf.transform([transformed_sms])
    # 3. predict
    result_sms = sms_model.predict(vector_input_sms)[0]
    # 4. Display
    if result_sms == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

email_cv = pickle.load(open('email_vectorizer.pkl', 'rb'))
email_model = pickle.load(open('email_model.pkl', 'rb'))        

st.title("Email Spam Classifier")

input_email = st.text_area("Enter the email")

if st.button('Predict Email'):
    # 1. preprocess
    transformed_email = transform_text(input_email)
    # 2. vectorize
    vector_input_email = email_cv.transform([transformed_email])
    # 3. predict
    result_email = email_model.predict(vector_input_email)[0]
    # 4. Display
    if result_email == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

fake_news_tfidf = pickle.load(open('fake_news_vectorizer.pkl', 'rb'))
fake_news_model = pickle.load(open('fake_news_model.pkl', 'rb'))        

st.title("Fake News Detector")

input_news = st.text_area("Enter the news or even the title of the news will work fine")

if st.button('Predict News'):
    # 1. preprocess
    transformed_news = transform_text(input_news)
    # 2. vectorize
    vector_input_news = fake_news_tfidf.transform([transformed_news])
    # 3. predict
    result_news = fake_news_model.predict(vector_input_news)[0]
    # 4. Display
    if result_news == 1:
        st.header("Fake News")
    else:
        st.header("Not a Fake News")