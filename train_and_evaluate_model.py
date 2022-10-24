import json
from transformers import AutoModel, AutoTokenizer, CamembertForSequenceClassification, CamembertForTokenClassification, CamembertTokenizer, CamembertForMaskedLM, CamembertModel, CamembertForCausalLM, CamembertConfig
import os
import numpy as np
from xml.etree import ElementTree as ET
import tensorflow as tf
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch.autograd.profiler as profiler
import gc
import sys
from collections import Counter
import nltk
from sentence_transformers import SentenceTransformer, util
from lemmagen3 import Lemmatizer
from sklearn.utils import shuffle


MAX_SEQ_LEN = 61
NUM_EPOCHS = 5




def read_ma_wsd_dict(filename):
    lem_sl = Lemmatizer('sl')
    ret_dict = {}
    with open(filename, 'r', encoding='utf-8') as f:
        X = []
        Y = []
        target_lemmas = []
        for i, line in enumerate(f):
            
            words = line.split('\t')
            target_lemma = words[0]
            target_pos = words[1]
            target_word = words[2]
            sense = words[3]
            location = words[4]
            sentence = words[5]
            X.append(sentence)
            Y.append(int(sense))
            target_lemmas.append(target_lemma.lower())
            if lem_sl.lemmatize(target_word) not in ret_dict.keys():
                ret_dict[lem_sl.lemmatize(target_word)] = [(sentence, location, sense)]
            else:
                ret_dict[lem_sl.lemmatize(target_word)].append((sentence, location, sense))
            

    return X, Y, target_lemmas, ret_dict
    
    
def read_ma_wsd_dict_tokenized_Y(filename):
    lem_sl = Lemmatizer('sl')
    ret_dict = {}
    with open(filename, 'r', encoding='utf-8') as f:
        X = []
        Y = []
        target_lemmas = []
        for i, line in enumerate(f):
            
            words = line.split('\t')
            target_lemma = words[0]
            target_pos = words[1]
            target_word = words[2]
            sense = words[3]
            location = words[4]
            sentence = words[5][:-1]
            X.append(sentence)
            Y_tokenized = [0 for _ in range(len(sentence.split(" ")))]
            
            Y_tokenized[int(location)] = int(sense)+1

            Y.append(Y_tokenized)
            
            

            target_lemmas.append(target_lemma.lower())
            if lem_sl.lemmatize(target_word) not in ret_dict.keys():
                ret_dict[lem_sl.lemmatize(target_word)] = [(sentence, location, sense)]
            else:
                ret_dict[lem_sl.lemmatize(target_word)].append((sentence, location, sense))
            

    return X, Y, target_lemmas, ret_dict
    



def read_elexis_wsd(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        X = []
        Y = []
        curr_sent_X = []
        curr_sent_Y = []
        for line in f:
            if line[0] == '#':
                continue
            if len(line) < 2:
                X.append(" ".join(curr_sent_X))
                Y.append(curr_sent_Y)
                curr_sent_X = []
                curr_sent_Y = []
                continue
            parts = line.split('\t')
            word = parts[1]
            lemma = parts[2]
            tags = parts[4].split('|')
            if len(tags) == 1:
                pos=None
                #sense_id=-100
                sense_id=0
            else:
                pos=tags[-2]
                sense_id=tags[-1]
            curr_sent_X.append(word)
            curr_sent_Y.append(int(sense_id))
            
        return X, Y
    
    
    
def expand_XY(X, Y):
    num_skipped = 0
    device = torch.device("cuda")
    slo_tokenizer = CamembertTokenizer.from_pretrained('./data/sloberta2')
    ret_X = []
    ret_Y = []
    target_positions = []
    for x, y in zip(X, Y):
        #for xx, yy in zip(x.split(" "), y):
        #    print(xx, yy)
        new_X = []
        new_Y = []
        i = 0
        tokens = slo_tokenizer(x, return_tensors = 'pt').to(device)
        words = slo_tokenizer.tokenize(x)
        if len(x.split(" ")) > 5:
            print(len(x.split(" ")), x.split(" ")[:4])
        if len(x.split(" ")) > 200:
            print(x)
            num_skipped += 1
            continue

        for word_y in y:

            while words[i][0] != '▁':

                new_Y.append(-100)
                i += 1

            new_Y.append(int(word_y))
            target_positions.append(i)
            i += 1
        del tokens
        del words
        while len(new_Y) < MAX_SEQ_LEN:
            new_Y.append(-100)
        if len(new_Y) > MAX_SEQ_LEN:
            new_Y = new_Y[:MAX_SEQ_LEN]
        ret_X.append(x)
        ret_Y.append(new_Y)
    print('SKIPPED', num_skipped)
    return ret_X, ret_Y, target_positions


def get_ma_dict_vectors(sent, location, sense, lemma, device, config, slo_tokenizer, encoder):
    lem_sl = Lemmatizer('sl')
    
    words = sent.split(" ")
    words_lemmatized = [lem_sl.lemmatize(w) for w in words]
    sent_lemmas = " ".join(words_lemmatized)
    tokens = slo_tokenizer(sent_lemmas, return_tensors = 'pt').to(device)
    
    words = slo_tokenizer.tokenize(sent_lemmas)

    if len(words) > 200:
        return None
    vectors = encoder(**tokens)[0].cpu().detach().numpy().tolist()[0]
    curr_pos = 0
    ret_vector = None
    for word, vector in zip(words, vectors):
        if word[0] == '▁':
            if curr_pos == int(location):
                ret_vector = vector
                curr_pos += 1
                break
            else:
                curr_pos += 1
    return ret_vector
    
    
    
def compare_to_ma_dict(sskj_sent, ma_dict, sskj_y, lem_sl):
    device = torch.device("cuda")
    config = CamembertConfig.from_pretrained('./data/sloberta2')
    slo_tokenizer = CamembertTokenizer.from_pretrained('./data/sloberta2')
    config.is_decoder=True
    encoder = CamembertForCausalLM.from_pretrained('./data/sloberta2', config=config)
    encoder.to(device = torch.device("cuda"))
    sskj_words = sskj_sent.split(" ")
    sskj_words_lemmas = [lem_sl.lemmatize(w) for w in sskj_words]
    sskj_sent_lemmas = " ".join(sskj_words_lemmas)
    tokens = slo_tokenizer(sskj_sent_lemmas, return_tensors = 'pt').to(device)
    words = slo_tokenizer.tokenize(sskj_sent_lemmas)
    vectors = encoder(**tokens)[0].cpu().detach().numpy().tolist()[0]
    i = 0
    curr_word = ""
    curr_vectors = []
    ret_senses = []
    sskj_i = 0
    while i < len(words): 
        if words[i][0] == '▁':
            if curr_word == "":
                curr_word = words[i]
                curr_vectors = [vectors[i]]
                i += 1
            else:
                is_in_dict = False
                if lem_sl.lemmatize(curr_word[1:]) in ma_dict.keys():
                    is_in_dict = True
                # If it's tagged in both
                if is_in_dict and sskj_y[sskj_i] != 0 :
                    max_sim = 0
                    max_sense = 0
                    for sent, location, sense in ma_dict[lem_sl.lemmatize(curr_word[1:])]:
                        ma_dict_vector = get_ma_dict_vectors(sent, location, sense, lem_sl.lemmatize(curr_word[1:]), device, config, slo_tokenizer, encoder)
                        if ma_dict_vector == None:
                            similarity = 0
                        else:
                            similarity = util.cos_sim(vectors[i], ma_dict_vector).detach().numpy().tolist()[0][0]
                        if similarity > max_sim:
                            max_sim = similarity
                            max_sense = int(sense) + 1

                    ret_senses.append(int(sense)+1)
                # Untagged in sskj, no point in tagging because we don't know the GT value. Can skip because evaluation code skips 0s anyway.
                if is_in_dict and sskj_y[sskj_i] == 0:
                    pass
                # Tagged in sskj, but can't find meanings in ma. We still need to predict something to match to GT. Predict -1 so that evaluation ignores it.
                if not is_in_dict and sskj_y[sskj_i] != 0:
                    ret_senses.append(-1)
                # Untagged, not in ma. Skip
                if not is_in_dict and sskj_y[sskj_i] == 0:
                    pass
                curr_word = words[i]
                curr_vectors = [vectors[i]]
                i += 1
                sskj_i += 1
        if i >= len(words):
            break
        while i < len(words) and words[i][0] != '▁':
            curr_word += words[i]
            curr_vectors.append(vectors[i])
            i += 1
    return ret_senses
        
    


combined_CAs_default = []
combined_CAs_skipped = []
#X_ma, Y_ma, _, ma_dict = read_ma_wsd_dict_tokenized_Y('./data/sense_examples.txt') 
X, Y = read_elexis_wsd('./data/elexis-wsd-sl_corpus.tsv')
max_cls = 0
total_default_correct = 0
total_skipped_correct = 0
total_skipped_length = 0
total_default_length = 0
current_sent_i = 0
lem_sl = Lemmatizer('sl')

X_new, Y_new, positions = expand_XY(X, Y)

#X_ma_new, Y_ma_new, positions_ma = expand_XY(X_ma, Y_ma)

max_len = 0
for y in Y_new:
    if len(y) > max_len:
        max_len = len(y)
for y in Y_ma_new:
    if len(y) > max_len:
        max_len = len(y)
print(max_len)




X_train, X_test, Y_train, Y_test  = train_test_split(X_new, Y_new, train_size=0.8, shuffle=True)


#print(len(X_train), len(X_test), len(Y_train), len(Y_test), len(X_ma_new), len(Y_ma_new))



X_train, Y_train = shuffle(X_train, Y_train)


print(X_train[0])
print(Y_train[0])

train_batches_X = [X_train[i:i + 32] for i in range(0, len(X_train), 32)]
train_batches_Y = [Y_train[i:i + 32] for i in range(0, len(Y_train), 32)]
test_batches_X = [X_test[i:i + 32] for i in range(0, len(X_test), 32)]
test_batches_Y = [Y_test[i:i + 32] for i in range(0, len(Y_test), 32)]

config = CamembertConfig.from_pretrained('./data/sloberta2')
config.is_decoder=True
model = CamembertForSequenceClassification.from_pretrained('./data/sloberta2', num_labels=28)
encoder = CamembertForCausalLM.from_pretrained('./data/sloberta2', config=config)
model_seq = CamembertForTokenClassification.from_pretrained('./data/sloberta2', num_labels=28)







model.train()
model_seq.train()
device = torch.device("cuda")
#device = torch.device("cpu")
model.to(device)
model_seq.to(device)
encoder.to(device)
#criterion = nn.CrossEntropyLoss()
#criterion.to(device)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model_seq.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model_seq.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-5)
#optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_seq.parameters()))
#slo_tokenizer = AutoTokenizer.from_pretrained('./sloberta2/')
slo_tokenizer = CamembertTokenizer.from_pretrained('./data/sloberta2')
#slo_tokenizer.to(device)

#model_seq.load_state_dict(torch.load('./model_ckpt_with_ma_data_zeros_instead_of_minus_100.ckpt'))
i = 0
running_loss = 0
for epoch in range(NUM_EPOCHS):
    model_seq.train()
    print('epoch', epoch)
    for batch_X, batch_Y in zip(train_batches_X, train_batches_Y):
        i += 1
        pt_batch = slo_tokenizer(
                    batch_X,
                    padding='max_length',
                    truncation=True,
                    max_length=MAX_SEQ_LEN,
                    return_tensors="pt").to(device)

        optimizer.zero_grad()
        #print(batch_Y)
        #print(type(batch_Y[0]))
        #print('batch Y', batch_Y)
        
        targets = torch.tensor(batch_Y).type(torch.LongTensor)
        targets = targets.to(device)
        #print(pt_batch)
        #print(targets)
        outputs = model_seq(**pt_batch, labels=targets)
        #print('outputs', outputs)
        del pt_batch
        del targets
        loss = outputs.loss
        loss.backward()
        running_loss += loss.item()
        if i % 30 == 0:
            print(i, 'loss', running_loss/30, file=sys.stderr)
            running_loss = 0
        optimizer.step()
        del loss
        gc.collect()
        torch.cuda.ipc_collect()
    torch.save(model_seq.state_dict(), './wsd_model.ckpt')

    model_seq.eval()
    CAs =  []
    combined_results = []
    all_preds = []
    with torch.no_grad():

        for i in range(1):
            loop_results = []
            #print(i)
            j = 0
            #for test_batch_X, test_batch_Y, direct_i, indirect_i in zip(test_batches_X, test_batches_Y, test_direct_i, test_indirect_i):
            for test_batch_X, test_batch_Y in zip(test_batches_X, test_batches_Y):
                test_encodings = slo_tokenizer(
                                    test_batch_X,
                                    padding='max_length',
                                    truncation=True,
                                    max_length=MAX_SEQ_LEN,
                                    return_tensors="pt").to(device)
                #print(test_encodings)
                preds = model_seq(**test_encodings)
                preds = preds.logits.tolist()
                #print(len(preds), len(preds[0]))
                loop_results += list(preds)
                #print(preds)
                #print(test_batch_Y)
                for sent_pred, sent_Y in zip(preds, test_batch_Y):
                    sent_pred = np.argmax(sent_pred, axis=1)
                    
                    filtered_pred = []
                    filtered_Y = []
                    for i in range(len(sent_Y)):
                        if sent_Y[i] >= 0:
                            filtered_pred.append(sent_pred[i])
                            filtered_Y.append(sent_Y[i])
                    filtered_pred = np.array(filtered_pred)
                    filtered_Y = np.array(filtered_Y)
                    #print(filtered_Y)
                    #print(filtered_pred)
                    #print(sent_pred == sent_Y)
                    #CAs.append(sum(sent_pred==sent_Y)/len(sent_pred))
                    CAs.append(sum(filtered_pred==filtered_Y)/len(filtered_pred))
                    all_preds.append(sent_pred)
                #print(i, j, '/', len(test_batches_X), file=sys.stderr)
                j += 1
            combined_results.append(loop_results)
            loop_results = []
            

    #print(combined_results)
    #print(CAs)

    #print(all_preds)
    print(np.mean(CAs))
    sentences = []
    document_pos_tags = []
    

model_seq.eval()
CAs =  []
combined_results = []
all_preds = []
with torch.no_grad():

    for i in range(1):
        loop_results = []
        #print(i)
        j = 0
        #for test_batch_X, test_batch_Y, direct_i, indirect_i in zip(test_batches_X, test_batches_Y, test_direct_i, test_indirect_i):
        for test_batch_X, test_batch_Y in zip(test_batches_X, test_batches_Y):
            test_encodings = slo_tokenizer(
                                test_batch_X,
                                padding='max_length',
                                truncation=True,
                                max_length=MAX_SEQ_LEN,
                                return_tensors="pt").to(device)
            #print(test_encodings)
            preds = model_seq(**test_encodings)
            preds = preds.logits.tolist()
            #print(len(preds), len(preds[0]))
            loop_results += list(preds)
            #print(preds)
            #print(test_batch_Y)
            for sent_pred, sent_Y in zip(preds, test_batch_Y):
                sent_pred = np.argmax(sent_pred, axis=1)
                
                filtered_pred = []
                filtered_Y = []
                for i in range(len(sent_Y)):
                    if sent_Y[i] >= 0:
                        filtered_pred.append(sent_pred[i])
                        filtered_Y.append(sent_Y[i])
                filtered_pred = np.array(filtered_pred)
                filtered_Y = np.array(filtered_Y)
                #print(filtered_Y)
                #print(filtered_pred)
                #print(sent_pred == sent_Y)
                #CAs.append(sum(sent_pred==sent_Y)/len(sent_pred))
                CAs.append(sum(filtered_pred==filtered_Y)/len(filtered_pred))
                all_preds.append(sent_pred)
            #print(i, j, '/', len(test_batches_X), file=sys.stderr)
            j += 1
        combined_results.append(loop_results)
        loop_results = []
        

#print(combined_results)
#print(CAs)

#print(all_preds)
print(np.mean(CAs))
sentences = []
document_pos_tags = []