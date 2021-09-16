import streamlit as st
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import pickle

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [c for c in text if c.isalnum()]
    inter = []
 
    for i in text:
        if i not in stopwords.words('english'):
            inter.append(i)
    text = inter[:]
    
    ps = PorterStemmer()
    inter.clear()
    
    for i in text:
        inter.append(ps.stem(i))
    
    text = inter[:]
    
    return " ".join(text)

tfidf = pickle.load(open('tfidf.pkl','rb'))
model = pickle.load(open('model_rf.pkl','rb'))

st.title("SMS Spam Classifier")

input = st.text_area("Enter the message")

if st.button('Predict'):
    #1.Preprocess
    transformed_text = transform_text(input)
    #2.Vectorize
    vector_input = tfidf.transform([transformed_text])
    print(vector_input)
    #3.Predict
    result = model.predict(vector_input)
    print(result)
    #4.Display
    if result[0] == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")