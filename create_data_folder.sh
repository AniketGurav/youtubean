#!/bin/bash
DATAPATH=$1

mkdir -p "$DATAPATH"/data/{word_embeddings,pickle,json,model,corpora}
mkdir -p "$DATAPATH"/data/corpora/opinion/{semeval-absa-2014,youtube}

mv Laptop_Train_v2.xml "$DATAPATH"/data/corpora/opinion/semeval-absa-2014/laptops_train.xml
mv Restaurants_Train_v2.xml "$DATAPATH"/data/corpora/opinion/semeval-absa-2014/restaurants_train.xml

mv ./deps.words.bz2 "$DATAPATH"/data/word_embeddings/deps.words.bz2
mv ./GoogleNews-vectors-negative300.bin.gz "$DATAPATH"/data/word_embeddings/GoogleNews-vectors-negative300.bin.gz
mv ~/senna/senna_embeddings.txt "$DATAPATH"/data/word_embeddings/senna_embeddings.txt

