# seq2seq

This repository contains a Deep Bi-directional encoder-decoder RNN that performs neural machine translation. The code is divided into 2 folders:
### Vanilla Model
- Ensure that you first create the data using `preprocess_vanilla.py`. After that, simply run `main.py` with `--version=vanilla`. 
- Hyperparameters used in the vanilla model: 
  - BATCH_SIZE = 32
  - EMBEDDING_SIZE = 512
  - RNN_SIZE = 256
  - NUMBER OF RNN LAYERS = 2
  - BIDIRECTIONALITY = TRUE
  - LEARNING_RATE = 2E-3
  - NUM_EPOCHS = 2
 - The above hyperparameters were used since they yielded the highest accuracy on the validation data while not taking too long to train on 1070 GPU.
 
 ### Byte-pair encoding model
 This version of the code pre-processes the corpus based on byte-pair encoding and then proceeds to train a deep seq2seq model. 
 - Ensure that you first create the data using `preprocess_bpe.py`. After that, simply run `main.py` with `--version=bpe`. 
 - Advantages of using byte-pair encoding: 
   - BPE allows us to improve translation for rarely seen words in our MT corpus. 
   - Even though we do not expect to see a big difference in overall accuracy while using BPE over vanilla data preprocessing, we expect that by breaking words into commanly seen sub-words, we will be able to better predict rarely seen words as a composition of more frequently seen sub-words. 
 
