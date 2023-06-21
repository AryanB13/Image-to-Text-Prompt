# Image-to-Text-Prompt
This project is used to generate text prompt from an image. It is trained using ResNet50 model, by using image vs prompt dataset. After training it can be used to generate text prompts for any image provided.
It uses Pytorch, Tensorflow, Keras to train the given dataset. After we provide a new image for text prompt prediction, it preprocesses it and outputs the embedding of generates prompt using 'paraphrase-MiniLM-L6-v2' model of Sentence Transformer as a csv file 'final.csv'.
