# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 20:53:23 2020

@author: Subham
"""
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten, Conv1D, MaxPooling1D, Embedding, Conv2D, MaxPool2D
from keras.models import Model

import keras.regularizers

def remove_punctuations(docs): #list of strings
    punctuations = '''!()[]{};:"\,<>'’‘‘./?@#$%^&*_~-–'''
    docs2=[];
    i=0;
    for sent in docs:
        s=""
        for x in sent:
            if(x not in punctuations):
                s=s+x;
            else:
                s=s+' '
        docs2.append(s);


    docs=docs2;
    docs2=[]
    return docs


def convert_lowercase(docs): #list of strings
    docs2=[]
    for sent in docs:
        sent=sent.lower()
        docs2.append(sent);

    docs=docs2;
    docs2=[]
    return docs




def tokenize_data(docs): #list of strings
    docs2=[]
    for sent in docs:
        sent=sent.split(' ')
        docs2.append(sent);

    docs=docs2;
    docs2=[]
    return docs

def remove_stopwords(docs): #list of list of words as param
    stop_words=["i", "br", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
    #stop_words=set(stopwords.words("english"));
    #stop_words.add('br')
    docs2=[]
    for sent in docs:
        s=[];
        for word in sent:
            if(word not in stop_words):
                s.append(word);
        docs2.append(s);

    docs=docs2;
    docs2=[]
    return docs


def make_string(docs): #list of list of words as param
    docs2=[]
    for sent in docs:
        s=""
        s=' '.join(sent)
        docs2.append(s);

    docs=docs2;
    docs2=[]
    return docs


def remove_digits(docs): #list of strings
    docs2=[]
    for sent in docs:
        sent2=sent.split(' ');
        words=[]
        for x in sent2:
            flag=0;
            for c in x:
                if(c>='0' and c<='9'):
                    flag=1;
                    break;
                else:
                    pass;
            if(flag==0):
                words.append(x);
        words=' '.join(words);
        docs2.append(words)
    docs=docs2;
    docs2=[]
    return docs


def filtertext(test_review):
    test_review_list=[]
    test_review_list.append(test_review)
    para_test=convert_lowercase(test_review_list)
    para_test=remove_digits(para_test)
    para_test=remove_punctuations(para_test)
    para_test=tokenize_data(para_test)
    para_test=remove_stopwords(para_test)
    para_test=make_string(para_test)
    return para_test