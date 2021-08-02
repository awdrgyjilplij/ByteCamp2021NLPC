import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset

def getTrainData(tokenizer):
    labels=[]
    input_ids=[]
    attn_mask=[]
    with open("../data/total.tsv","r",encoding="utf-8") as f:
        for line in tqdm(list(f)[1:100]):
            line=line.strip().split('\t')
            labels.append(int(line[1]))
            tokenized_data=tokenizer(line[4],padding='max_length',truncation=True,max_length=512)
            input_ids.append(tokenized_data["input_ids"])
            attn_mask.append(tokenized_data["attention_mask"])
            
    labels=torch.tensor(labels, dtype=torch.long)
    input_ids=torch.tensor(input_ids, dtype=torch.long)
    attn_mask=torch.tensor(attn_mask, dtype=torch.long)
    dataset=TensorDataset(input_ids, attn_mask, labels)
    return dataset

if __name__ == '__main__':
    pass