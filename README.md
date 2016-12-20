Code and dataset for the paper "Mining opinions on closed-captions of Youtube videos with RNNs"

- Download the needed data and software
    1. download and install Senna, http://ronan.collobert.com/senna/
    2. download and install CoreNLP 3.6, http://stanfordnlp.github.io/CoreNLP/history.html
    3. download GoogleNews embeddings from https://code.google.com/archive/p/word2vec/
    4. download WikiDeps embeddings from https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/
    5. download and unzip the SemEval2014 V2 Train data from http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools

- Create environment
    1. create_data_folder.sh path_where_to_create_data_folder
    2. modify youtubean/enlp/settings.py accordingly

- Prepare model training data
    1. prepare_best_models.sh restaurants|laptops|samsung|weighted|pred

- Run models
    1. run_best_models.sh restaurants|laptops|samsung|weighted|pred