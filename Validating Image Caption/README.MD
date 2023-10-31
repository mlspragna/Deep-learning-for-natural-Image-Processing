# Consistency Assessment of Image and Caption

This project is designed to create a system that determines the degree of consistency between an image and its associated caption. The system combines the power of Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN)/Transformer, and Feedforward Neural Networks (FFNN) to provide a consistent verdict.

[Overview](#Overview)

The goal of this project is to build a deep neural network model that takes an image and its associated caption as input and outputs a value between 0 and 1, indicating the degree of consistency. A value of 1 implies a high degree of consistency between the image and caption, while 0 implies inconsistency.

## Project Structure

The project can be divided into several stages:

1. **Data Collection and Preprocessing**:
   - Gather a dataset of image-caption pairs, labeled as consistent or inconsistent.
   - Preprocess the data, including image resizing and caption tokenization.

2. **Stage 1: Image Processing (CNN)**:
   - Use a pre-trained CNN model(we have used InceptionV3) to extract meaningful image features.
   - Fine-tune the CNN model if necessary to adapt to the specific task.
   - Output: Image feature vector.

3. **Stage 2: Caption Processing (RNN/Transformer)**:
   - Utilize an RNN (LSTM/GRU) or Transformer model to process tokenized captions.
   - Generate a caption feature vector that encodes semantic information.
   
4. **Stage 3: Combining Features (FFNN)**:
   - Concatenate the image and caption feature vectors.
   - Use a Feedforward Neural Network (FFNN) to learn the relationship between the image and caption features.
   - The FFNN outputs a value between 0 and 1, indicating consistency.


## Project Contributors
   - Muthyala Leela Sri Pragna
   - Nikitha Narakulla
   - Vineetha Vennela
