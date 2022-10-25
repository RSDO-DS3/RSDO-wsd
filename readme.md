V tem repozitoriju se nahaja rezultat aktivnosti A3.3 - R3.2.3 Orodje za ekstrakcijo povezav, ki je nastalo v okviru projekta Razvoj slovenščine v digitalnem okolju.

---

This repository contains the code for the WSD api. It is split into two main components

1. The code used for the training and evaluation of the WSD model (located in `./train_and_evaluate_model.py`)
2. The code used to create the docker container with the WSD api. (located in `./app/api.py`)

`./data/` contains the data necessary for training and evaluating the model.

# Training and evaluating the model
train_and_evaluate_model.py contains the code for training and evaluating the model. Running the file will train and evaluate a model on data from the elexis-wsd-sl_corpus.tsv (https://www.clarin.si/repository/xmlui/handle/11356/1674), located in ./data/. We use a Camembert token prediction model to train the WSD model (Martin, Louis, et al. "CamemBERT: a Tasty French Language Model." ACL 2020-58th Annual Meeting of the Association for Computational Linguistics. 2020.). The model currently achieves a classification accuracy of .45 when evaluated on the elexis-wsd-sl test set and we are currently in the process of improving the model to achieve better results.

By default, the model requires pytorch with GPU acceleration using CUDA (https://pytorch.org/get-started/locally/).

If you want to train a model on your own data, replace the elexis-wsd-sl_corpus.tsv files with your own data.

# Building the WSD api docker container
The code for the WSD api is located in ./app/api.py. To build the docker container, place the required model files into ./data. This will create a container running the api using the uvicorn server. The container requires three files:
	
1. The pretrained sloberta2 model (https://www.clarin.si/repository/xmlui/handle/11356/1397), which should be placed inside ./data/sloberta2
2. A trained wsd model, which should be placed inside ./data and named wsd_model.ckpt
3. A sense inventory. Currently, we use elexis-wsd-sl_sense-inventory.tsv, which should be placed inside ./data

You can then build the container by running `docker build -t rsdo_wsd .`.  To start the API container, use the command `docker run --gpus all -d --name rsdo_wsd_container -p 80:80 rsdo_wsd`.

# Running the api locally
To run the API locally: 

1. Extract the contents of this repository.
2. Download the requirements using `pip install -r requirements.txt`
3. Run the API server using uvicorn app.api:app --host 127.0.0.1 --port 80

This will start the server using the ip 127.0.0.1 and port 80. The api accepts POST requests at /predict/wsd. The endpoint requires two parameters:

1. `inventory`. This specifies the sense inventory which contains all possible word definitions. It should contain a string with the inventory name.  Currently, only "DSB" is supported
2. `text`. This should contain a string whith the text to be disambiguated 

The api will return the following for each word:

1. `sense_id`. The id of the identified word sense
2. `inventory`. The sense inventory passed as the parameter
3. `lemma`. The lemma of the word
4. `definition`. The definition of identified word sense
5. `score`. The confidence of the obtained prediction. Higher numbers indicate higher confidence.

To test the server, try sending a POST request using curl:

`curl -X POST -H "accept:application/json" -H "Content-Type:application/json" -d "{\"inventory\": \"DSB\", \"text\": \"Soba ima dvoje vrat.\" }" http://127.0.0.1:80/predict/wsd -L`



---
> Operacijo Razvoj slovenščine v digitalnem okolju sofinancirata Republika Slovenija in Evropska unija iz Evropskega sklada za regionalni razvoj. Operacija se izvaja v okviru Operativnega programa za izvajanje evropske kohezijske politike v obdobju 2014-2020.
