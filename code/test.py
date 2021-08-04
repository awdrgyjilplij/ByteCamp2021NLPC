import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
import pickle
import logging
from transformers import BertForSequenceClassification
from torch.utils.data import SequentialSampler, DataLoader


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def accuracy(preds, labels):
    sum1 = sum([1 if preds[i] == labels[i] and labels[i]
               == 1 else 0 for i in range(len(preds))])
    sum2 = sum([1 if preds[i] == labels[i] else 0 for i in range(len(preds))])
    return sum2/len(preds), sum1/(sum(preds)+1e-6), sum1/(sum(labels)+1e-6)


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device %s n_gpu %d distributed training", device, n_gpu)

    pretrained = "bert-base-chinese"
    state_dict_ = torch.load("model/model_best.pt")
    state_dict = {}
    for key in state_dict_.keys():
        state_dict[key[7:]] = state_dict_[key]
    model = BertForSequenceClassification.from_pretrained(
        pretrained, state_dict=state_dict)

    torch.cuda.empty_cache()
    model.to(device)
    model = torch.nn.DataParallel(model)
    model.eval()
    print("model "+pretrained+" params: ",
          sum([param.nelement() for param in model.parameters()]))

    labels = []
    ids = []
    preds = []
    preds_score = []

    dataset = pickle.load(open("data/unlabel_features.pkl", 'rb'))
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler,
                             batch_size=1024, drop_last=False)

    with torch.no_grad():
        for batch in tqdm(list(data_loader)):
            batch = tuple(t.to(device) for t in batch)
            input_ids, attn_mask, labels_, ids_ = batch

            logits = model(input_ids=input_ids,
                           attention_mask=attn_mask, labels=labels_)[1]
            logits = nn.Softmax(dim=1)(logits)
            logits = logits.cpu().numpy()
            ids += ids_.squeeze(-1).cpu().numpy().tolist()
            preds += np.argmax(logits, axis=1).tolist()
            preds_score += logits.tolist()
            labels += labels_.squeeze(-1).cpu().numpy().tolist()

    print(accuracy(preds, labels))

    with open("result.txt", "w", encoding="utf-8") as f:
        for i in range(len(ids)):
            f.write("%d %d %.3f %.3f\n" %
                    (ids[i], preds[i], preds_score[i][0], preds_score[i][1]))


if __name__ == '__main__':
    main()
