########## IMPORTS ##########
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModel, AutoTokenizer, CamembertForSequenceClassification, CamembertForTokenClassification, CamembertTokenizer, CamembertForMaskedLM, CamembertModel, CamembertForCausalLM, CamembertConfig
import numpy as np
from lemmagen3 import Lemmatizer


def get_definition(lemma, sense_id):
    definition = ""
    if lemma in sense_dict.keys():
        definitions = sense_dict[lemma]
        if str(sense_id) in definitions.keys():
            definition = definitions[str(sense_id)]
        else:
            definition = "Definicija ni bila najdena"
    else:
        definition = "Definicija ni bila najdena"
    return definition
                
    


   

########## INPUT OBJECT ##########
class InputJSON(BaseModel):
    inventory: str
    text: str
    
########## MODELS ##########
device = torch.device("cpu")
config = CamembertConfig.from_pretrained('./data/sloberta2')
model_seq = CamembertForTokenClassification.from_pretrained('./data/sloberta2', num_labels=28)
model_seq.load_state_dict(torch.load('./data/wsd_model.ckpt', map_location=torch.device('cpu')))
model_seq.to(device)

lem_sl = Lemmatizer('sl')

slo_tokenizer = CamembertTokenizer.from_pretrained('./data/sloberta2')


########## SENSE DICT ##########
with open('./data/elexis-wsd-sl_sense-inventory.tsv', 'r', encoding='utf-8') as f:
    sense_dict = {}
    for i, line in enumerate(f):
        if i == 0:
            continue
        parts = line.split("\t")
        lemma = parts[0]
        sense_tags = parts[2]
        sense_id = parts[2].split("|")[-1]
        definition = parts[3]
        if lemma not in sense_dict.keys():
            sense_dict[lemma] = {sense_id: definition}
        else:
            sense_dict[lemma][sense_id] = definition

#print(len(preds), len(preds[0]))


#print('Preloading models...')

#model_folder = '../models/'

# read "models" folder for models to serve
#model_names = [name for name in os.listdir(model_folder) if os.path.isdir(os.path.join(model_folder, name))]

# or manually define the models to serve
#model_names = [
#   'sloberta-squad2-SLO',
#   'xlm-roberta-base-squad2-SLO'
#]

"""
models = {}
tokenizers = {}

for model_name in model_names:
    print(f'Loading {model_name}...')
    model = AutoModelForQuestionAnswering.from_pretrained(f'../models/{model_name}') 
    tokenizer = AutoTokenizer.from_pretrained(f'../models/{model_name}') 

    models[model_name] = model
    tokenizers[model_name] = tokenizer

print('Models loaded!')

"""
########## API ##########
print('loading api')

app = FastAPI()

print('loaded')
@app.get("/")
async def status():
    return {"status": "active"}

@app.post("/predict/wsd/")
async def qa(input: InputJSON):
    sent = input.text
    inventory = input.inventory
    MAX_SEQ_LEN = 61
    
    test_encodings = slo_tokenizer([sent],
                               padding='max_length',
                               truncation=True,
                               max_length=MAX_SEQ_LEN,
                               return_tensors="pt").to(device)
    #print(test_encodings)
    preds = model_seq(**test_encodings)
    preds = torch.nn.functional.softmax(preds.logits, dim=2).tolist()
    sent_pred = np.argmax(preds[0], axis=1)
    sent_score = np.max(preds[0], axis=1)
    words = slo_tokenizer.tokenize(sent)
    ret_list = []
    curr_word = ""
    curr_preds = []
    curr_scores = []
    for i in range(len(words)):
        print(words[i], sent_pred[i])
        if words[i][0] == '▁':
            if curr_word == "":
                curr_word = words[i]
                curr_preds.append(sent_pred[i])
                curr_scores.append(sent_score[i])
            else:
                lemma = lem_sl.lemmatize(curr_word[1:])
                ret_list.append({"sense_id": int(curr_preds[0]), 
                                 "word": curr_word[1:],
                                 "inventory": inventory,
                                 "score": float(curr_scores[0]),
                                 "definition": get_definition(lemma, curr_preds[0]),
                                 "lemma": lemma})
                curr_word = words[i]
                curr_preds = [sent_pred[i]]
                curr_scores = [sent_score[i]]
        else:
            if words[i] not in [',', '.', ':', ';', '-', '(', ')']:
                curr_word += words[i]
                curr_preds.append(sent_pred[i])
                curr_scores.append(sent_score[i])
        
    if curr_word != "":
        lemma = lem_sl.lemmatize(curr_word[1:])
        ret_list.append({"sense_id": int(curr_preds[0]), 
                                 "word": curr_word[1:],
                                 "inventory": inventory,
                                 "score": float(curr_scores[0]),
                                 "definition": get_definition(lemma, curr_preds[0]),
                                 "lemma": lemma})



    return ret_list
