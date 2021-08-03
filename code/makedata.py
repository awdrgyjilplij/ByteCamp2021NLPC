import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset

def getTrainData(tokenizer):
    labels=[]
    input_ids=[]
    attn_mask=[]
    with open("data/total_new.tsv","r",encoding="utf-8") as f:
        for line in tqdm(list(f)[10000:]):
            try:
                line=line.strip().split('\t')
                labels.append(int(line[1]))
                text=line[4].split('<')[0]
                tokenized_data=tokenizer(text,padding='max_length',truncation=True,max_length=512)
                input_ids.append(tokenized_data["input_ids"])
                attn_mask.append(tokenized_data["attention_mask"])
            except:
                pass
            
    labels=torch.tensor(labels, dtype=torch.long)
    input_ids=torch.tensor(input_ids, dtype=torch.long)
    attn_mask=torch.tensor(attn_mask, dtype=torch.long)
    dataset=TensorDataset(input_ids, attn_mask, labels)
    return dataset

def getEvalData(tokenizer):
    labels=[]
    input_ids=[]
    attn_mask=[]
    with open("data/total_new.tsv","r",encoding="utf-8") as f:
        for line in tqdm(list(f)[1:10000]):
            try:
                line=line.strip().split('\t')
                labels.append(int(line[1]))
                text=line[4].split('<')[0]
                tokenized_data=tokenizer(text,padding='max_length',truncation=True,max_length=512)
                input_ids.append(tokenized_data["input_ids"])
                attn_mask.append(tokenized_data["attention_mask"])
            except:
                pass
            
    labels=torch.tensor(labels, dtype=torch.long)
    input_ids=torch.tensor(input_ids, dtype=torch.long)
    attn_mask=torch.tensor(attn_mask, dtype=torch.long)
    dataset=TensorDataset(input_ids, attn_mask, labels)
    return dataset

if __name__ == '__main__':
    pass