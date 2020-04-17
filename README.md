# nlp-project


#### Abstract

In this project, we train word embeddings on a corpus consisting of speeches given in the Finnish parliament. We use transcripts from the years 2008--2016.  
If time allows we will also use data from earlier years based on already parsed records by one of the authors. We compare the performance of three to five of the models: word2vec, FastTEXT, Elmo, BERT and GloVe.  

We evaluate the performance of these models based on semantic similarity judgment resource Finsemeval and on a small set of self-constructed analogical reasoning tasks, like ``Keskusta - Juha Sipilä + Petteri Orpo = Kokoomus''. We also test how a model trained on the political speech corpus performs on a more general corpus (Finnish Wikipedia text corpus).   

Data available at [https://korp.csc.fi/download/eduskunta/v1/](https://korp.csc.fi/download/eduskunta/v1/)


### fastText

Installation: [https://fasttext.cc/docs/en/supervised-tutorial.html](https://fasttext.cc/docs/en/supervised-tutorial.html)