from tqdm import tqdm
import random

data=[]
with open("data/total_new.tsv","r",encoding="utf-8") as f:
    for i,line_ in tqdm(list(enumerate(list(f)[1:]))):
        try:
            line=line_.strip().split('\t')
            label=int(line[2])
            if label==1:
                data.append(line_)
            else:
                if i%40==0:
                    data.append(line_)
        except:
            pass

random.shuffle(data)
with open("data/total_sample.tsv","w",encoding="utf-8") as f:
    for line in data:
        f.write(line)
