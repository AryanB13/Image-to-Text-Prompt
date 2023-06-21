import numpy as np
import pandas as pd
import cv2
import os
from glob import glob

images_path='/home/icps/Desktop/pmt/kaggle/stable-diffusion-image-to-prompts/images/'
images=glob(images_path+'*.png')
len(images)

from keras.applications import ResNet50

incept_model=ResNet50(include_top=True)


from keras.models import Model
last=incept_model.layers[-2].output
modele=Model(inputs=incept_model.input,outputs=last)
modele.summary()

def img_process(i):
    img = cv2.imread(i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    img = img.reshape(1,224,224,3)
    return img


images_features = {}
prompt_embeddings_inp = []
count = 0
for i in images:
    img=img_process(i)
    pred = modele.predict(img).reshape(2048,).flatten()
    
    prompt_embeddings_inp.append(pred)
        
    img_name = i.split('/')[-1]
    images_features[img_name] = pred
    
    count += 1
    
#     if count > 7:
#         break
        
#     elif count <=7812:
#         print(count)



prompt_embeddings_inp = np.array(prompt_embeddings_inp)

#Textpreprocess

prompt_path='/home/icps/Desktop/pmt/kaggle/stable-diffusion-image-to-prompts/prompts.csv'

import csv

prompts = []
with open(prompt_path, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        prompts.append(row)


prompt_dict_in = {}
for i in prompts[1:]:
    img_name = i[0] + ".png" 
    prompt = i[1]
    prompt_dict_in[img_name] = [prompt]
#     if img_name in images_features:
#         if img_name not in prompt_dict_in:
#             prompt_dict_in[img_name] = [prompt]
               
#         else:
#             prompt_dict_in[img_name].append(prompt)


def preprocessed(txt):
    modified='startofseq '+txt[0]+' endofseq'
    return modified

for k,v in prompt_dict_in.items():
    prompt_dict_in[k] =preprocessed(v)


count_words = {}
count=1
for k,v in prompt_dict_in.items():
    for word in v.split():
        if word not in count_words:
            count_words[word] = count
            count+=1


for k, v in prompt_dict_in.items():
    encoded = []
    for word in v.split():  
        encoded.append(count_words[word])
    prompt_dict_in[k] = encoded



from keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 0
for k, v in prompt_dict_in.items():
    if len(v) > MAX_LEN:
        MAX_LEN = len(v)
        print(v)


VOCAB_SIZE = len(count_words)
def generator(photo, prompt):    
    X = []    #2048 vals
    y_in = []   #pre string
    y_out = []  #predicted    
    for k, v in prompt.items():
        X.append(photo[k])
        in_seq= [v[:1]]
        out_seq = v[1]
        in_seq = pad_sequences(in_seq, maxlen=MAX_LEN, padding='post', truncating='post')[0]
        out_seq = to_categorical([out_seq], num_classes=VOCAB_SIZE+1)[0]
        y_in.append(in_seq)
        y_out.append(out_seq)            
    return X, y_in, y_out

X, y_in, y_out = generator(images_features, prompt_dict_in)


X = np.array(X)
y_in = np.array(y_in, dtype='float64')
y_out = np.array(y_out, dtype='float64')



from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers import add
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from keras.models import Sequential, Model




embedding_size = 128
max_len = MAX_LEN
vocab_size = len(count_words)+1

image_model = Sequential()

image_model.add(Dense(embedding_size, input_shape=(2048,), activation='relu'))
image_model.add(RepeatVector(max_len))

image_model.summary()

language_model = Sequential()

language_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
language_model.add(LSTM(256, return_sequences=True))
language_model.add(TimeDistributed(Dense(embedding_size)))

language_model.summary()

conca = Concatenate()([image_model.output, language_model.output])
x = LSTM(128, return_sequences=True)(conca)
x = LSTM(512, return_sequences=False)(x)
x = Dense(vocab_size)(x)
out = Activation('softmax')(x)
model = Model(inputs=[image_model.input, language_model.input], outputs = out)

# model.load_weights("../input/model_weights.h5")
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
model.summary()





model.fit([X, y_in], y_out, batch_size=512, epochs=50)


inv_dict = {v:k for k, v in count_words.items()}



from sentence_transformers import SentenceTransformer
modelsent = SentenceTransformer('paraphrase-MiniLM-L6-v2')

prompt_embeddings_out = []

count = 0
prompt = ''
prompt_out = {}
breakout=0

prompt_dict_out={}

for i in images:
    img=img_process(i)

    test_feature = modele.predict(img)

    text_inp = ['startofseq']
    #print(i)
    
    embedding=[]
    count = 0

    while count < 30:
        count += 1
        encoded = []
        for j in text_inp:
            encoded.append(count_words[j])
        encoded = [encoded]

        encoded = pad_sequences(encoded, padding='post', truncating='post', maxlen=MAX_LEN)

        prediction = model.predict([test_feature, encoded])
        #print(prediction)
        
        
        prompt_embed = model.predict([test_feature, encoded])
        
#         prompt_embedding = modelsent.encode([prompt])[0]
#         embedding.append(prompt_embedding)
        
        embedding.append(prompt_embed[0][count - 1])  # Assume the output embedding is at the current count index
        #print(embedding)
        
        
        sampled_word = inv_dict[np.argmax(prediction)]
        
        if sampled_word == 'endofseq':
            break

        prompt = prompt + ' ' + sampled_word

        text_inp.append(sampled_word)
        
        
    #print(embedding)

    img_name = i.split('/')[-1]
    prompt_dict_out[img_name]=prompt
    
    prompt_embedding = modelsent.encode([prompt])
    prompt_embedding = prompt_embedding.flatten()
    prompt_embeddings_out.append(prompt_embedding)

    
    breakout+=1
#     if breakout==200:
#         break




brk=0
prompt_final=[]
check=[]

import csv
output_file = 'final.csv'

with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["imgId_eId", "val"]) 
    for i in images:    
        img_name = i.split('/')[-1]
        img_name = img_name.split('.')[0]
        #print(img_name)
        j=0
        #print(brk)
        while (True):
            imgId_eId = f"{img_name}_{j}"
            eId_in = prompt_embeddings_inp[brk][j]
            
            eId_out = prompt_embeddings_out[brk][j]
            
            writer.writerow([imgId_eId, eId_out])
            if j==len(prompt_embeddings_out[brk])-1:
                break
            j+=1
            
            #print(eId_out)
            
            
        brk += 1
#         prompt_final.append((img_name, prompt_dict_out[img_name]))
        #print(brk)
#         if brk == 199:
#             break
