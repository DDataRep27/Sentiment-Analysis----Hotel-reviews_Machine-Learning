#ML-App for sentiment analysis - hotel reviews!!

#Import Libraries
import pandas as pd
import streamlit as st 
from pickle import load 
#from PIL import Image
import time
from streamlit_lottie import st_lottie, st_lottie_spinner
import json

#Background config
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
    
lottie_start = load_lottiefile("19146-tapered-hello.json")
lottie_type  = load_lottiefile("74424-worker-have-an-idea.json")
lottie_process = load_lottiefile("72689-brain-bulb-with-gears.json")
    
    
#Input variables 
def user_input_features():
    X = st.text_area(label='Play around with our sentiment analyzer:', placeholder=('Type your text here and check the sentiment!!'))

    new = {'Review': X}

    features = pd.DataFrame(new, index=[0])

    return features


#Loading the model
model = load(open('hrsa_intelligence.joblib', 'rb'))

   
#Image on webpage
#img = Image.open("back1.jpg")
#st.image(img)


#Home Webage
if not st.checkbox("Let's get started!!"):
    st_lottie(lottie_start, height=400, width=None, quality="high", speed=1.1, loop=True)


#Sentiment Analysis page
else: 
    
#Title of web page
    st.title("Sentiment Analyzer")

    st_lottie(lottie_type, height=190, width=200, quality="high", speed=1.4, loop=True)

    df = user_input_features()

#Model Prediction
    result = model.predict(df)

#Predicting the final result
    if st.button('Classify Text'):
        with st_lottie_spinner(lottie_process, height=(225), width=(700), quality="med", speed=1.25):
            time.sleep(1.6)
            st.subheader('Predicted Result:')
        if result[0]==0:
            st.error("Negative Statement", icon='😡')    
        
        else:
            st.success('Positive Statement', icon='😊')
  
       
if __name__=='__user_input_features__':
    user_input_features()







