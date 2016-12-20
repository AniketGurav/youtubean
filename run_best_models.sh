#!/bin/bash

SEED=42
DATA_PATH=$1
MODEL=$2
LOGFILE=$3

############# Restaurants ##################

if [ "$MODEL" == "restaurants" ]; then
  python train.py "$DATA_PATH"/json/[SennaChunker]SemEval14RestaurantsTrain.WikiDeps.json "$DATA_PATH"/model JordanRNN 5 3 100 6 0.01 "$SEED" -f -ft -d -v --log "$LOGFILE"

  python train.py "$DATA_PATH"/json/[SennaChunker]SemEval14RestaurantsTrain.WikiDeps.json "$DATA_PATH"/model LSTM 5 3 100 6 0.01 "$SEED" -f -ft -d -v --log "$LOGFILE"

  python train.py "$DATA_PATH"/json/[SennaChunker]SemEval14RestaurantsTrain.WikiDeps.json "$DATA_PATH"/model ElmanBRNN 5 1 100 6 0.01 "$SEED" -ft -d -v --log "$LOGFILE"

  python train.py "$DATA_PATH"/json/[SennaChunker]SemEval14RestaurantsTrain.SennaEmbeddings.json "$DATA_PATH"/model ElmanRNN 5 3 200 6 0.01 "$SEED" -f -ft -d -v --log "$LOGFILE"

  python train.py "$DATA_PATH"/json/[SennaChunker]SemEval14RestaurantsTrain.WikiDeps.json "$DATA_PATH"/model BLSTM 5 1 200 6 0.01 "$SEED" -f -ft -d -v --log "$LOGFILE"
fi

############# Laptops ##################

if [ "$MODEL" == "laptops" ]; then
  python train.py "$DATA_PATH"/json/[CoreNLPConstParser.CoNLL2000Chunker]SemEval14LaptopsTrain.SennaEmbeddings.json "$DATA_PATH"/model JordanRNN 5 3 50 6 0.01 "$SEED" -ft -d -v --log "$LOGFILE"

  python train.py "$DATA_PATH"/json/[CoreNLPConstParser.CoNLL2000Chunker]SemEval14LaptopsTrain.SennaEmbeddings.json "$DATA_PATH"/model LSTM 5 3 100 6 0.01 "$SEED" -f -ft -d -v --log "$LOGFILE"

  python train.py "$DATA_PATH"/json/[CoreNLPConstParser.CoNLL2000Chunker]SemEval14LaptopsTrain.SennaEmbeddings.json "$DATA_PATH"/model ElmanBRNN 5 1 50 6 0.01 "$SEED" -f -ft -d -v --log "$LOGFILE"

  python train.py "$DATA_PATH"/json/[CoreNLPConstParser.CoNLL2000Chunker]SemEval14LaptopsTrain.SennaEmbeddings.json "$DATA_PATH"/model ElmanRNN 5 3 50 6 0.01 "$SEED" -f -ft -d -v --log "$LOGFILE"

  python train.py "$DATA_PATH"/json/[CoreNLPConstParser.CoNLL2000Chunker]SemEval14LaptopsTrain.SennaEmbeddings.json "$DATA_PATH"/model BLSTM 5 1 50 6 0.01 "$SEED" -f -ft -d -v --log "$LOGFILE"
fi

############# SamsungGalaxyS5 ##################

if [ "$MODEL" == "samsung" ]; then

  python train.py "$DATA_PATH"/json/[SennaChunker]SamsungGalaxyS5.WikiDeps "$DATA_PATH"/model ElmanRNN 5 3 100 6 0.01 "$SEED" -f -ft -d -v --log "$LOGFILE"

  python train.py "$DATA_PATH"/json/[CoreNLPConstParser.CoNLL2000Chunker]SamsungGalaxyS5.WikiDeps "$DATA_PATH"/model ElmanRNN 5 3 200 6 0.01 "$SEED" -ft -d -v --log "$LOGFILE"

  python train.py "$DATA_PATH"/json/[CoreNLPConstParser.CoNLL2000Chunker]SamsungGalaxyS5.SennaEmbeddings "$DATA_PATH"/model LSTM 5 3 100 6 0.01 "$SEED" -ft -d -v --log "$LOGFILE"

  python train.py "$DATA_PATH"/json/[CoreNLPConstParser.CoNLL2000Chunker]SamsungGalaxyS5.WikiDeps "$DATA_PATH"/model ElmanBRNN 5 1 200 6 0.01 "$SEED" -ft -d -v --log "$LOGFILE"

  python train.py "$DATA_PATH"/json/[SennaChunker]SamsungGalaxyS5.SennaEmbeddings "$DATA_PATH"/model BLSTM 5 1 100 6 0.01 "$SEED" -f -ft -d -v --log "$LOGFILE"
fi

############# WEIGHTED ##################

if [ "$MODEL" == "weighted" ]; then
  python train.py "$DATA_PATH"/json/WEIGHTED[SennaChunker]SemEval14LaptopsTrain.[SennaChunker]SamsungGalaxyS5.GoogleNews "$DATA_PATH"/model LSTM 5 3 50 6 0.01 "$SEED" -f -ft -d -v --log "$LOGFILE"

  python train.py "$DATA_PATH"/json/WEIGHTED[SennaChunker]SemEval14LaptopsTrain.[SennaChunker]SamsungGalaxyS5.GoogleNews "$DATA_PATH"/model ElmanRNN 5 3 100 6 0.01 "$SEED" -f -ft -d -v --log "$LOGFILE"

  python train.py "$DATA_PATH"/json/WEIGHTED[SennaChunker]SemEval14LaptopsTrain.[SennaChunker]SamsungGalaxyS5.WikiDeps "$DATA_PATH"/model ElmanRNN 5 3 100 6 0.01 "$SEED" -f -ft -d -v --log "$LOGFILE"
fi

############# PRED ##################

if [ "$MODEL" == "pred" ]; then
  python train.py "$DATA_PATH"/json/PRED[CoreNLPConstParser.CoNLL2000Chunker]SemEval14LaptopsTrain.[SennaChunker]SamsungGalaxyS5.SennaEmbeddings "$DATA_PATH"/model LSTM 5 3 100 6 0.01 "$SEED" -f -ft -d -v --log "$LOGFILE"

  python train.py "$DATA_PATH"/json/PRED[SennaChunker]SemEval14RestaurantsTrain.[SennaChunker]SamsungGalaxyS5.WikiDeps "$DATA_PATH"/model BLSTM 5 1 100 6 0.01 "$SEED" -f -ft -d -v --log "$LOGFILE"

  python train.py "$DATA_PATH"/json/PRED[SennaChunker]SemEval14RestaurantsTrain.[SennaChunker]SamsungGalaxyS5.WikiDeps "$DATA_PATH"/model ElmanBRNN 5 1 100 6 0.01 "$SEED" -f -ft -d -v --log "$LOGFILE"

  python train.py "$DATA_PATH"/json/PRED[SennaChunker]SemEval14RestaurantsTrain.[SennaChunker]SamsungGalaxyS5.WikiDeps "$DATA_PATH"/model ElmanBRNN 5 1 200 6 0.01 "$SEED" -f -ft -d -v --log "$LOGFILE"

  python train.py "$DATA_PATH"/json/PRED[CoreNLPConstParser.CoNLL2000Chunker]SemEval14LaptopsTrain.[SennaChunker]SamsungGalaxyS5.SennaEmbeddings "$DATA_PATH"/model ElmanRNN 5 3 100 6 0.01 "$SEED" -f -ft -d -v --log "$LOGFILE"

  python train.py "$DATA_PATH"/json/PRED[CoreNLPConstParser.CoNLL2000Chunker]SemEval14LaptopsTrain.[SennaChunker]SamsungGalaxyS5.SennaEmbeddings "$DATA_PATH"/model JordanRNN 5 3 200 6 0.01 "$SEED" -f -ft -d -v --log "$LOGFILE"
fi
