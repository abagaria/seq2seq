# seq2seq

This repository contains a Deep Bi-directional encoder-decoder RNN that performs neural machine translation. The code is divided into 2 folders:

## Hyperparameters
I used the same hyperparameters for all my models
--num_rnn_layers=1
--bidirectional=True
--batch_size=32 
--epochs=1 
--embedding_size=256 
--rnn=512 
--lr=1e-3

I tried adding more layers to my RNN, but that didnt help much with accuracy.
I reduced my learning rate from the one I used to train my vanilla seq2seq model. This is because the training loss was decreasing too quickly.
To prevent the loss from falling too quickly, I want to try SGD optimizer in the future. 

## French -> English Translation
Validation perplexity = 18.74
Validation accuracy = 44.5%

## German -> English Translation
Validation perplexity = 18.89
Validation accuracy = 43.8%

## French + German -> English Translation:
Validation perplexity = 15.29
Validation accuracy = 46.5%

I get a slight improvement over the vanilla seq2seq model (~42%) but the accuracy isn't close to what I got with the BPE model on French -> English in the previous assignment (~53%).
My hypothesis is that in this assignment our vocabulary size exploded - where we had a vocab size of ~10,000 with mono-lingual vocab translations, the combined word piece vocab when using all 3 languages is closer to 60,000.
As a result of the much larger vocab size, the probability of incorrectly classifying a word should go up. 

We get a 2% bump in classification accuracy when training on data which goes from French AND German to English. This suggests that training on 2 input languages perhaps has a regularizing impact on the language model for english. Since we get examples for a French sentence and a German sentence that correspond to the same english translation, we get a regularizing effect while learning a language model over that english sentence.
