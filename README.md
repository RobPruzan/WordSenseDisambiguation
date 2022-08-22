# WordSenseDisambiguation
**BERT for Word Sense Disambiguation**

Utilizing the last hidden state embedding of BERT corresponding to the target word and potential word sense, the 2 word embeddings are compared with cosine similarity, 
if that value passes a threshold, we deem the word senses equivelenet 

The target word and potential word sense embeddings are computed in seperate forward passes, but both need to be inputted with contextual information (a sentence) for 
self-attention enrinched embeddings to represent the word-sense inside the state 

Related-Work: Huang et al., 2020 | GlossBERT: BERT for Word Sense Disambiguation with Gloss Knowledge | Association for Computational Linguistics
https://arxiv.org/pdf/1908.07245.pdf 
