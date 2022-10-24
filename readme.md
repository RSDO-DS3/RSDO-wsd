This repository contains the code for the WSD api. It is split into two main components

1. The code used for the training and evaluation of the WSD model.
2. The code used to create the docker container with the WSD api.

# Training and evaluating the model
train_and_evaluate_model.py contains the code for training and evaluating the model. Running the file will train and evaluate a model on data from and elexis-wsd-sl_corpus.tsv (https://www.clarin.si/repository/xmlui/handle/11356/1674), located in ./data/. We use a Camembert token prediction model to train the WSD model (Martin, Louis, et al. "CamemBERT: a Tasty French Language Model." ACL 2020-58th Annual Meeting of the Association for Computational Linguistics. 2020.)

By default, the model requires pytorch with GPU acceleration using CUDA (https://pytorch.org/get-started/locally/).

If you want to train a model on your own data, replace the elexis-wsd-sl_corpus.tsv files with your own data.

# Building the WSD api docker container
The code for the WSD api is located in ./app/api.py. To build the docker container, place the required model files into ./data. This will create a container running the api using the uvicorn server. The container requires three files:
	
1. The pretrained sloberta2 model (https://www.clarin.si/repository/xmlui/handle/11356/1397), which should be placed inside ./data/sloberta2
2. A trained wsd model, which should be placed inside ./data and named wsd_model.ckpt
3. A sense inventory. Currently, we use elexis-wsd-sl_sense-inventory.tsv, which should be placed inside ./data
