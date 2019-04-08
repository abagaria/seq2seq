# seq2seq

This repository contains a Deep Bi-directional encoder-decoder RNN that performs neural machine translation. The code is divided into 2 folders:

## Hyperparameters
I used the same hyperparameters for all my models
--num_rnn_layers=1
--bidirectional=True
--batch_size=32 
--epochs=1 
--embedding_size=512 
--rnn=512 
--lr=1e-3

I tried adding more layers to my RNN, but that didnt help much with accuracy.
I reduced my learning rate from the one I used to train my vanilla seq2seq model. This is because the training loss was decreasing too quickly.
To prevent the loss from falling too quickly, I want to try SGD optimizer in the future. 

## English -> French Translation
Validation perplexity = 21

Validation accuracy = 42.6%

## English ->  German Translation
Validation perplexity = 23.92

Validation accuracy = 41.9%

##  English -> French + German Translation:
Validation perplexity = 25.65

Validation accuracy = 40.5%

## German -> French Zeroshot Translation
Validation perplexity = 92.35

Validation accuracy = 29.9%

### One to many translation
My accuracy for the one-many translation model was not better than the monolingual translation model as I had expected. It performs pretty much the same as the monolingual MT models.

### Zero-shot translation reproduction
I was able to emulate the zero-shot translation results from the paper. This was essentially because of the "implicit bridging" phenomenon described in the paper. By training on French -> English and English -> German and using the target token during translation, the model develops a strong prior on how to translate between French -> German

### Data pre-processing verification
- I wrote out the data being used at every stage of the pipeline to a text file and verified that it made sense
- I also printed out the input to the model (decoded using a reverse vocabulary) to ensure that the data into and out of the model made sense.

### Model and training verification
- My guess as to why i dont get a high enough accuracy is that I have not implemented any form of attention
- I have verified that the model is implemented correctly by going through it line by line
- I visualized the loss function using tensorboard and made sure that it was decreasing
- I tried a few different set of hyperparameters. I did not perform a grid search over them as I don't see any educational value in doing that. 
- To improve accuracy numbers, I would experiment with implementing attention and doing a more extensive hyperparameter search

