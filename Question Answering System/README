# Question Answering System with RNN

This repository contains an implementation of a Question Answering (QA) system using a Recurrent Neural Network (RNN). The system is designed to answer FACTOID-type questions (WHO, WHAT, WHEN, WHERE) by extracting answers from a given context paragraph. It employs an encoder-decoder network and has been trained on the SQuAD dataset.

## Introduction

The Question Answering system is designed to answer questions by extracting single-word or phrase answers from a given context paragraph. It uses a bidirectional LSTM (Bi-LSTM) based encoder-decoder architecture, along with an attention mechanism for context understanding.

### Dataset

The system is trained on the SQuAD dataset, which is a widely recognized benchmark for QA systems. It includes a collection of context paragraphs and corresponding factoid questions with single-word or phrase answers. This system focuses on answering WHO, WHAT, WHEN, and WHERE questions.

For more information on the SQuAD dataset, please refer to [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/).

### Architecture

The architecture of the QA system consists of the following components:

- **Encoder**: It comprises two Bi-LSTM models, one for the context and one for the paragraph. The context and paragraph embeddings are fed into these models to capture contextual information.

- **Attention Layer**: This layer calculates attention scores over the hidden states in the paragraph to generate a context vector, which is a weighted average of the paragraph's hidden states. This context vector is then concatenated with the encoded paragraph.

- **Decoder**: The decoder model is a Bi-LSTM followed by a dense layer and a softmax activation. It predicts the start and end indices of the answers within the paragraph.
