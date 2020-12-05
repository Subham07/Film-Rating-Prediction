# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 21:21:24 2020

@author: Subham
"""
import tensorflow as tf
import preprocess
from flask import Flask, request, jsonify, render_template

import pickle
import numpy as np

app=Flask(__name__)
model_deep=pickle.load(open('deep_model.pkl','rb'))
global graph
graph = tf.get_default_graph()
glove_embedding=pickle.load(open('glove_emb.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text=str(request.form.get("review"))
    print("text: ",text)
    test_review_list=[]
    test_review_list.append(text)
    para_test=preprocess.convert_lowercase(test_review_list)
    para_test=preprocess.remove_digits(para_test)
    para_test=preprocess.remove_punctuations(para_test)
    para_test=preprocess.tokenize_data(para_test)
    para_test=preprocess.remove_stopwords(para_test)
    para_test=preprocess.lemmatize(para_test)
    para_test=preprocess.make_string(para_test)
    vectors=preprocess.vectorize(para_test,glove_embedding)
    with graph.as_default():
        test_pred=model_deep.predict_proba(np.reshape(vectors,(1,100)))
    return render_template('index.html',prediction_text='{:0.2f}/5'.format(test_pred[0][0]*5))
    
    
if __name__ == "__main__":
    app.run(debug=True)

'''
text="On the whole, director Hansal Mehta gets the setting quite right with a lot of detailing. But with more conviction in execution and writing, this one could've hit it out of the park."

final_text=predict(text)
print(final_text)
'''