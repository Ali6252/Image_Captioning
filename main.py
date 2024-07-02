
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import keras
import json
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import InceptionV3
import os
import streamlit as st
from keras.models import Model
import warnings

warnings.filterwarnings("ignore")


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'



def caption_generator(photo):
    global max_length,idx_to_word
    image=photo.reshape(1,2048)
    model = keras.saving.load_model("Models/image_caption_model_3.keras")
    in_text="start"
    for i in range(max_length):
        L=[]
        for j in in_text.split(" "):
            file=open("Data/glove.6B.200d.txt",encoding="utf-8") #every time we are reading because after looping through text file pointer will be at end when you loop again it will start at end of file 
            try:
                for line in file:
                    values = line.split()
                    word=values[0]
                    if word==j:
                        L.append(np.asarray(values[1:], dtype="float32").tolist())
                        break
            except:
                pass
        seq=pad_sequences([L],maxlen=max_length,dtype="float32")
        pred=model.predict([image,seq],verbose=0)
        index=np.argmax(pred)
        next_word=idx_to_word[str(index)]
        if next_word == "end":
            break
        in_text=in_text+" "+next_word
    return in_text[6:]

def read_image():
    image_file = st.file_uploader(" ",type=["jpg", "png"])
    if image_file:
        img = load_img(image_file).resize([299,299])
        st.write("The Image you uploaded is \n")
        st.image(img)
        img = img_to_array(img)
        img = preprocess_input(img)
        image_model = InceptionV3(weights="imagenet")
        image_model = Model(inputs=image_model.inputs, outputs=image_model.layers[-2].output)
        img2vec=image_model.predict(img.reshape(-1, 299, 299, 3), verbose=0).flatten()
        st.write("The Caption is\n")
        st.write(caption_generator(img2vec))
    else:
        st.write("Please upload image")


if __name__=="__main__":

    st.title("Image Captioning")
    st.write("This Model is Trained on flickr8k images please dont upload any other images and except the exact result you can epect approimate reults please be generous")
    st.write("please find the dataset of images from below link")
    url="https://www.kaggle.com/datasets/adityajn105/flickr8k"
    st.write(url)
    f = open("Data/idx_to_word.json")
    idx_to_word = json.load(f)
    max_length = 34  # thi we get from training data max length of sentence

    read_image()
