from unicodedata import category
import torch
from tqdm import tqdm
import random
from torch.utils.data import TensorDataset
import pickle
import csv
import os
from transformers import BertTokenizer
import pandas as pd

def getData(type,tokenizer):
    labels=[]
    input_ids=[]
    ids=[]
    title_ids=[]
    text_ids=[]
    categories=[]
    attn_mask=[]
    if type == "all":
        df = pd.read_csv('data/reli_pseduo_data.csv', sep='\t')
        df = df.values.tolist()
        random.shuffle(df)
        for line in tqdm(df):
            try:
                # line=line.strip().split('\t')
                ids.append([int(line[0])])
                labels.append([int(line[1])])
                text=line[4].split('<')[0]
                tokenized_data=tokenizer(text,padding='max_length',truncation=True,max_length=512)
                input_ids.append(tokenized_data["input_ids"])
                attn_mask.append(tokenized_data["attention_mask"])
                # categories.append([int(line[2])])
                # title=line[3]
                # text_ids.append(tokenizer(text)["input_ids"])
                # title_ids.append(tokenizer(title)["input_ids"])
            except:
                pass
    if type in [0,1,2,3]:
        ith=0
        with open("data/reli_train_%d.tsv"%type,"r",encoding="utf-8") as f:
            for line in tqdm(list(f)[1:]):
                try:
                    ith+=1
                    line=line.strip().split('\t')
                    # if int(line[1])==0 and ith%6!=0:
                    #     continue
                    labels.append([int(line[1])])
                    # categories.append([int(line[2])])
                    text=line[4]
                    # title=line[3]
                    tokenized_data=tokenizer(text,padding='max_length',truncation=True,max_length=512)
                    input_ids.append(tokenized_data["input_ids"])
                    attn_mask.append(tokenized_data["attention_mask"])
                    # text_ids.append(tokenizer(text)["input_ids"])
                    # title_ids.append(tokenizer(title)["input_ids"])
                except:
                    pass
    if type=="train":
        for i in range(3):
            with open("data/reli_train_%d.tsv"%i,"r",encoding="utf-8") as f:
                for line in tqdm(list(f)[1:]):
                    try:
                        line=line.strip().split('\t')
                        labels.append([int(line[1])])
                        text=line[4].split('<')[0]
                        tokenized_data=tokenizer(text,padding='max_length',truncation=True,max_length=512)
                        input_ids.append(tokenized_data["input_ids"])
                        attn_mask.append(tokenized_data["attention_mask"])
                    except:
                        pass
    if type=="eval":
        with open("data/reli_train_3.tsv","r",encoding="utf-8") as f:
            for line in tqdm(list(f)[1:]):
                try:
                    line=line.strip().split('\t')
                    labels.append([int(line[1])])
                    text=line[4].split('<')[0]
                    tokenized_data=tokenizer(text,padding='max_length',truncation=True,max_length=512)
                    input_ids.append(tokenized_data["input_ids"])
                    attn_mask.append(tokenized_data["attention_mask"])
                except:
                    pass

    # ids=torch.tensor(ids, dtype=torch.long)
    labels=torch.tensor(labels, dtype=torch.long)
    input_ids=torch.tensor(input_ids, dtype=torch.long)
    # title_ids=torch.tensor(title_ids, dtype=torch.long)
    # text_ids=torch.tensor(text_ids, dtype=torch.long)
    # categories=torch.tensor(categories, dtype=torch.long)
    attn_mask=torch.tensor(attn_mask, dtype=torch.long)
    # print(labels.shape,input_ids.shape,attn_mask.shape)
    dataset=TensorDataset(input_ids, attn_mask, labels)
    # dataset={"title_ids":title_ids,"text_ids":text_ids,"categories":categories,"labels":labels}
    return dataset

def getTrainData(tokenizer):
    feature_file = "data/train_features.pkl"
    if os.path.exists(feature_file):
        train_dataset = pickle.load(open(feature_file, 'rb'))
    else:
        train_dataset = getData("train",tokenizer)
        with open(feature_file, 'wb') as w:
            pickle.dump(train_dataset, w)
    return train_dataset

def getEvalData(tokenizer):
    feature_file = "data/eval_features.pkl"
    if os.path.exists(feature_file):
        eval_dataset = pickle.load(open(feature_file, 'rb'))
    else:
        eval_dataset = getData("eval",tokenizer)
        with open(feature_file, 'wb') as w:
            pickle.dump(eval_dataset, w)
    return eval_dataset

def getQuatData(tokenizer):
    datasets=[]
    for i in range(4):
        feature_file = "data/%d_features.pkl"%i
        if os.path.exists(feature_file):
            dataset = pickle.load(open(feature_file, 'rb'))
        else:
            dataset = getData(i,tokenizer)
            with open(feature_file, 'wb') as w:
                pickle.dump(dataset, w)
        datasets.append(dataset)
    return datasets

if __name__ == '__main__':
    feature_file = "data/pseduo_features.pkl"
    dataset=getData("all",BertTokenizer.from_pretrained("bert-base-chinese"))
    with open(feature_file, 'wb') as w:
        pickle.dump(dataset, w)
