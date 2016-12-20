#!/bin/bash

DATA_PATH=$1
MODEL=$2

############# Restaurants ##################

if [ "$MODEL" == "restaurants" ]; then
  python process_corpus.py --corpus SemEval14RestaurantsTrain --annotators Senna
  python generate_json.py "$DATA_PATH"/json --corpus SemEval14RestaurantsTrain --embeddings WikiDeps SennaEmbeddings
fi

############# Laptops ##################

if [ "$MODEL" == "laptops" ]; then
  python process_corpus.py --corpus SemEval14LaptopsTrain --annotators CoreNLP Senna
  python generate_json.py "$DATA_PATH"/json --corpus SemEval14LaptopsTrain --embeddings SennaEmbeddings
fi

############# SamsungGalaxyS5 ##################

if [ "$MODEL" == "samsung" ]; then
  python process_corpus.py --corpus SamsungGalaxyS5 --annotators Senna CoreNLP
  python generate_json.py "$DATA_PATH"/json --corpus SamsungGalaxyS5 --embeddings WikiDeps SennaEmbeddings --folds 10
fi

############# WEIGHTED ##################

if [ "$MODEL" == "weighted" ]; then
  python weighted_generate_json.py "$DATA_PATH"/json --source SemEval14LaptopsTrain --target SamsungGalaxyS5 --embeddings WikiDeps GoogleNews --folds 10
fi

############# PRED ##################

if [ "$MODEL" == "pred" ]; then
  python pred_generate_json.py "$DATA_PATH"/json create --source SemEval14LaptopsTrain --target SamsungGalaxyS5 --embeddings SennaEmbeddings
  python pred_generate_json.py "$DATA_PATH"/json create --source SemEval14RestaurantsTrain --target SamsungGalaxyS5 --embeddings WikiDeps

  python train.py "$DATA_PATH"/json/PRED[CoreNLPConstParser.CoNLL2000Chunker]SemEval14LaptopsTrain.[SennaChunker]SamsungGalaxyS5.SennaEmbeddings.json "$DATA_PATH"/model ElmanRNN 5 3 50 6 0.01 123 -ft -f -d -v
  python train.py "$DATA_PATH"/json/PRED[SennaChunker]SemEval14RestaurantsTrain.[SennaChunker]SamsungGalaxyS5.WikiDeps.json "$DATA_PATH"/model LSTM 5 3 100 6 0.01 123 -ft -f -d -v

  python pred_generate_json.py "$DATA_PATH"/json add "$DATA_PATH""/model/PRED[CoreNLPConstParser.CoNLL2000Chunker]SemEval14LaptopsTrain.[SennaChunker]SamsungGalaxyS5.SennaEmbeddings/bs=6 decay=True feat=True fine_tuning=True lr=0.01 ne=5 nh=50 rnn_type=ElmanRNN seed=123 sentence_train=False weight=1.0 ws=3" --folds 10
  python pred_generate_json.py "$DATA_PATH"/json add "$DATA_PATH""/model/PRED[SennaChunker]SemEval14RestaurantsTrain.[SennaChunker]SamsungGalaxyS5.WikiDeps/bs=6 decay=True feat=True fine_tuning=True lr=0.01 ne=5 nh=100 rnn_type=LSTM seed=123 sentence_train=False weight=1.0 ws=3" --folds 10
fi

