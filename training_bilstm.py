# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 14:25:55 2020

@author: Subham
"""

import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras
import tensorflow as tf
import pandas as pd
import preprocess_wo_nltk as preprocess
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.layers import Dropout
from sklearn.feature_extraction.text import CountVectorizer
max_words=20000


#importing from dataset
df=pd.read_csv('IMDB Dataset.csv')

mx=len(df['review'][0].split(' '))
posa=0
mn=len(df['review'][0].split(' '))
posb=0
for i in range(50000):
    if(len(df['review'][i].split(' ')) > mx):
        mx=len(df['review'][i].split(' '))
        posa=i;
    elif(len(df['review'][i].split(' ')) < mn):
        mn=len(df['review'][i].split(' '))
        posb=i;
print("maximum length: ",mx)
print("minimum length: ",mn)

reviews=df['review'].values
print(len(reviews))
targets=np.where(df['sentiment'].values=='positive',1,0)
print(len(targets))
print(targets)

#filtering text
para_test=preprocess.convert_lowercase(reviews)
para_test=preprocess.remove_digits(para_test)
para_test=preprocess.remove_punctuations(para_test)
para_test=preprocess.tokenize_data(para_test)
para_test=preprocess.remove_stopwords(para_test)
#para_test=preprocess.lemmatize(para_test)
para_test=preprocess.make_string(para_test)


#calculating mean length of reviews
lens=0
for i in range(50000):
    lens=lens+len(para_test[i].split(' '))
    
print("mean length: ", lens/50000) #120

#count distinct words
vocab={}
for i in range(50000):
    for words in para_test[i].split(' '):
        vocab[words]=1

print("Total Distinct words: ",len(vocab)) #approx 90K


#using tokenizer
vect=Tokenizer()
vect.fit_on_texts(para_test)
vocab_size = len(vect.word_index) + 1
print(vocab_size)


print(vect.word_index['stop'])

pickle.dump(vect, open('text_ind_2.pkl', 'wb'))

#encoded_docs

encoded_docs_train = vect.texts_to_sequences(para_test)
max_length = vocab_size
maxlen=150
padded_docs_train = pad_sequences(encoded_docs_train, maxlen=150, padding='post', truncating='post')
print(padded_docs_train.shape)

#train and test set
import numpy as np
X_final=np.array(padded_docs_train)
y_final=np.array(targets)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


#defining the model
embedding_vector_features=20
model1=Sequential()
model1.add(Embedding(vocab_size,output_dim=embedding_vector_features,input_length=maxlen))
model1.add(Bidirectional(LSTM(25)))
model1.add(Dropout(0.5))
model1.add(Dense(1,activation='sigmoid'))
model1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model1.summary())


# Training
model1.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=4,batch_size=256)

import pickle
with open('deep_bilstm_model_2.pkl', 'wb') as handle2:
    pickle.dump(model1, handle2, protocol=pickle.HIGHEST_PROTOCOL)
    

    
        
    
#testing new string
    
review_test="Sharib Hashmi turns out a stirring performance as the domestic worker who never quits being one. He makes the viewers empathise with his character Raicharan, especially in the sequence where he breaks down. Sharad Kelkar is fantastic as usual. Furthermore, with limited screen time, Rasika Dugal as Raichu’s wife Bhuri makes her presence felt. Harsh Chhaya and Flora Saini lend their support in pushing the narrative forward. All said, ‘Darbaan’ is an unusual story of loyalty, friendship, caregiving and the ultimate sacrifice that is powered by superlative performances, which make it a noteworthy tale. Definitely, a must watch."

test_review_list=[]
test_review_list.append(review_test)
para_test=preprocess.convert_lowercase(test_review_list)
para_test=preprocess.remove_digits(para_test)
para_test=preprocess.remove_punctuations(para_test)
para_test=preprocess.tokenize_data(para_test)
para_test=preprocess.remove_stopwords(para_test)
para_test=preprocess.make_string(para_test)

print(para_test)
test_vectors=vect.texts_to_sequences(para_test)

padded_docs_test = pad_sequences(test_vectors, maxlen=150, padding='post', truncating='post')
print(padded_docs_test.shape)


test_pred=model1.predict_proba(np.reshape(padded_docs_test,(1,150)))
print("prediction_rating='{:0.2f}/5'".format(test_pred[0][0]*5))


review_test_2="Debutant Vignarajan (who has also written the story) has a solid theme in hand – four different people, four different stories, all connected by a single thread. Each character is neatly written and has enough meat. The actors, too, have sunk their teeth into their parts with much gusto. Arjun showcases both his machismo and vulnerability with ease – he’s aggressive in the scenes in which he desperately wants to get away from the caller, and cowers with fear when he realises that he may have encountered something paranormal. Vinoth impresses as a visually impaired librarian – he is effortless while dealing with Braille and totally looks the part. Pooja and Kumar Natarajan have played their parts subtly, but leave behind a deep impression. Edwin Sakay’s camera (the first few frames help establish the mood of the film beautifully), Rembon Balraj’s art production (Arjun’s single-room house is straight out of mafia movies, but the red set-up makes it eerie. And oh, those rotary dial telephones are quite striking, we must say) and Pradeep Kumar’s music enrich the narration. But there’s something about the packaging that doesn't quite sit right – it’s the thread that connects the stories together. It’s flimsy, it's loose, but you realise that only when you reach the climax. At 2 hours 51 minutes, the film is too long. Though the narration keeps you glued, a lacklustre twist makes you question if your patience was worth it. The film claims to be a supernatural thriller, but there’s no spook in the story. Psychological elements? Aplenty. But the chill in the story is lost somewhere between wanting to be both."

test_review_list=[]
test_review_list.append(review_test_2)
para_test=preprocess.convert_lowercase(test_review_list)
para_test=preprocess.remove_digits(para_test)
para_test=preprocess.remove_punctuations(para_test)
para_test=preprocess.tokenize_data(para_test)
para_test=preprocess.remove_stopwords(para_test)
para_test=preprocess.make_string(para_test)

print(para_test)
test_vectors=vect.texts_to_sequences(para_test)

padded_docs_test = pad_sequences(test_vectors, maxlen=150, padding='post', truncating='post')
print(padded_docs_test.shape)


test_pred=model1.predict_proba(np.reshape(padded_docs_test,(1,150)))
print("prediction_rating='{:0.2f}/5'".format(test_pred[0][0]*5))



review_test_3="He is a self-proclaimed ‘hard-core lover’ and there are very few things in life Asif wouldn’t do for his spouse of three years. It is this very reason that prompts him to pack his bags and rush off to Daman to patch things up between Rashmi and her family, and maybe procure a place in their hearts for him, too.“He runs granite and marbles business and is a senior member at the Jagoo Aavam committee; they are fighting the superstitious beliefs surrounding ghosts and spirits,” declares Rashmi – proudly – when quizzed about her husband’s profession. So when the neighbourhood kids refuse to play at the adjacent ground because an evil spirit dwells there, Asif, quite expectedly, brushes it off and pledges to play a full IPL game on that abandoned land. Soil nailing, very dramatically, changes the Earth above and beneath – dark, ferocious clouds looming around at an alarming pace, while dried autumn leaves caress the faces. The witch sleeping underneath that piece of turf has been shaken awake, and now, she must follow the culprit home: Asif. Too dramatic? So is the plot of this film, read on. When the spirit does make her way into that home, it’s complete mayhem and madness. Who is it? What does she want? And why is Asif having a little escapade of his own with sarees and bangles? Questions aplenty, answers lie in ‘Laxmii’ – literally and figuratively. What makes or breaks a comedy are its one-liners, the actors’ timing and the relevance of the jokes and where they land. Unfortunately, Akshay Kumar-Kiara Advani starrer ’Laxmii’ has none. The project is ambitious and has cherry-picked an interesting angle to inculcate into the narrative – of the ghost being a transgender with unfinished business – but the execution is over the top and melodramatic even by comedy standards. The first half is laden with random bickering among characters that are flat-out pointless. Masala movie-making man Farhad Samji’s adaptive screenplay has many a loophole and there are inconsistencies that would be criminal to look past. Case in point: If Kiara’s parents are celebrating their 25th anniversary and her character has eloped three years back, how is an evidently much older Deepak (Manu Rishi Chadha) her elder brother? Samji tries hard to hit the bull’s eye with this one by conjuring up chaos at peak moments, but with a cattle farmer being referred to as ‘Hi, cow man! Andar aao,’ there wasn’t much hope whatsoever. Was there? Akshay Kumar’s quite the charmer as a staunch atheist and then a man grappling with demonic possession, but then again, when is he not? And Kiara Advani balances his act out with her sensible demeanour and that pretty smile. Rajesh Sharma as Papa is a henpecked husband trying to convince everyone – and himself really – that he is in charge of his life. But with a tanker of a wife (Ayesha Raza Mishra) and a jagrata-crazed son, Deepak (Manu Rishi Chadha), Sharma portrays the role of an inactive player in denial with comfort. Ayesha Raza Mishra and Ashwini Kalsekar’s Ashwini do all the screaming and hollering; not a single scene squeezes even a mild grin out of the viewers; if that was the intent at all. Raghava Lawrence’s direction feels like an effort to overcompensate for the lack of clarity and humour in its plot; the discomfort and need to polish up is visible and quite obvious. The music rendered by Tanishk Bagchi, Anup Kumar and Shashi-Khushi, though dropped at random intervals, is hummable. Peppy tunes like ‘Burjkhalifa’ and ‘BamBholle’ resonate for their quirkiness (good quirk, that is!). However, the same cannot be said about Amar Mohile’s background score – tepid and misplaced. ‘Laxmii’ sets out to be a satire against age-old beliefs and biases – we get it! – but the insipidness of the narrative and whatever follows thereon, butchers the lessons it originally desires to impart. Looking back, the reiterating catchphrase happens to be ‘Maine kuch nahin dekha… yahaan koi nahin hain’ and that’s the vibe we’re latching on to at the moment."

test_review_list=[]
test_review_list.append(review_test_3)
para_test=preprocess.convert_lowercase(test_review_list)
para_test=preprocess.remove_digits(para_test)
para_test=preprocess.remove_punctuations(para_test)
para_test=preprocess.tokenize_data(para_test)
para_test=preprocess.remove_stopwords(para_test)
para_test=preprocess.make_string(para_test)

print(para_test)
test_vectors=vect.texts_to_sequences(para_test)

padded_docs_test = pad_sequences(test_vectors, maxlen=150, padding='post', truncating='post')
print(padded_docs_test.shape)


test_pred=model1.predict_proba(np.reshape(padded_docs_test,(1,150)))
print("prediction_rating='{:0.2f}/5'".format(test_pred[0][0]*5))