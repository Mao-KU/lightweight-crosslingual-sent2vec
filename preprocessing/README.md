## English - French
Unzip "data.zip" and you can find the dataset file for English-French.
path: paracrawl/paracrawl.enfr.shuf.lower.clean
There also exists a file called "enfr.test" for you to test the code.

## English - German, English - Spanish, English - Italian
For other langauge pairs, we will release the cleaned dataset on Github in future.

## Other Details
1. BPE: process raw texts using BPE and get processed texts and the corresponding vocabulary. And the processes contain 1. BPE to get tokens 2. add [PAD] and [MAKSED] into vocabulary. 
- Tokenization: split sentences into words
- BPE: using words to apply BPE
2. Dataset: generate one sample including the ids, lengths for each sentences, masked token and the id for this token.
3. Dataloder: generate batch_size samples based on sample generated from Dataset

