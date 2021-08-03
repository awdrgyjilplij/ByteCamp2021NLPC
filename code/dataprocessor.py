import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset
import pickle
import os

def getData(type,tokenizer):
    labels=[]
    input_ids=[]
    attn_mask=[]
    with open("data/total_sample.tsv","r",encoding="utf-8") as f:
        f=list(f)[1000:] if type=="train" else list(f)[:1000]
        for line in tqdm(f):
            try:
                line=line.strip().split('\t')
                labels.append([int(line[2])])
                text=line[5].split('<')[0]
                tokenized_data=tokenizer(text,padding='max_length',truncation=True,max_length=512)
                input_ids.append(tokenized_data["input_ids"])
                attn_mask.append(tokenized_data["attention_mask"])
            except:
                pass

    labels=torch.tensor(labels, dtype=torch.long)
    input_ids=torch.tensor(input_ids, dtype=torch.long)
    attn_mask=torch.tensor(attn_mask, dtype=torch.long)
    # print(labels.shape,input_ids.shape,attn_mask.shape)
    dataset=TensorDataset(input_ids, attn_mask, labels)
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

if __name__ == '__main__':
    pass