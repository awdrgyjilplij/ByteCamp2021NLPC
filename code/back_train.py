import torch
import torch.nn as nn
import os
import argparse
from tqdm import tqdm
import numpy as np
import logging
import pickle
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from dataprocessor import getQuatData, getTrainData, getEvalData
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import SequentialSampler, DataLoader, RandomSampler

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def accuracy(logits,labels):
    outputs = np.argmax(logits, axis=1)
    sum1 = sum([1 if outputs[i]==labels[i] and labels[i]==1 else 0 for i in range(len(outputs))])
    return sum(outputs == labels)/len(outputs),sum1/sum(outputs),sum1/sum(labels)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_ids",
                        default='0,1,2,3,4,5,6,7',
                        type=str)
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int)
    parser.add_argument("--eval_batch_size",
                        default=256,
                        type=int)
    parser.add_argument("--a_dropout_prob",
                        default=0.1,
                        type=float)
    parser.add_argument("--h_dropout_prob",
                        default=0.1,
                        type=float)
    parser.add_argument("--s_dropout_prob",
                        default=0.1,
                        type=float)
    parser.add_argument("--warmup_prop",
                        default=0.1,
                        type=float)
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float)
    parser.add_argument("--num_train_epochs",
                        default=8,
                        type=int)
    parser.add_argument('--seed',
                        type=int,
                        default=42)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device %s n_gpu %d distributed training",device, n_gpu)

    pretrained="hfl/chinese-bert-wwm-ext"
    model_config = BertConfig.from_pretrained(
        pretrained, attention_probs_dropout_prob=args.a_dropout_prob, hidden_dropout_prob=args.h_dropout_prob,
        summary_last_dropout=args.s_dropout_prob)

    model = BertForSequenceClassification.from_pretrained(pretrained, config=model_config)
    tokenizer = BertTokenizer.from_pretrained(pretrained)

    torch.cuda.empty_cache()
    model.to(device)
    model = torch.nn.DataParallel(model)
    print("model "+pretrained+" params: ",sum([param.nelement() for param in model.parameters()]))

    train_dataset = pickle.load(open("data/pseduo_features.pkl", 'rb'))
    datasets = getQuatData(tokenizer)
    evalset = datasets[3]
    eval_sampler = SequentialSampler(evalset)
    eval_loader = DataLoader(evalset, sampler=eval_sampler, batch_size=args.eval_batch_size, drop_last=False)
    train_sampler = SequentialSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, drop_last=False)

    step_per_epoch = len(train_loader)
    total_steps = step_per_epoch*args.num_train_epochs
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = total_steps*args.warmup_prop, 
                                            num_training_steps = total_steps)
    best_accuracy = 0
    logits = []
    label_ids = []
    logger.info("***** Running training *****")
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", step_per_epoch)

    for ie in range(int(args.num_train_epochs)):
        model.train()
        train_loss = 0
        with tqdm(total=step_per_epoch, desc='Epoch %d' % (ie + 1)) as pbar:
            for batch in train_loader:
                batch = tuple(t.to(device) for t in batch)
                input_ids, attn_mask, labels, ids = batch

                loss = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)[0]
                if n_gpu > 1:
                    loss = loss.mean()  
                train_loss+=loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()  
                scheduler.step()
                model.zero_grad()
                pbar.set_postfix({'loss': "%.3f"%loss})
                pbar.update(1)

        model.eval()
        eval_loss = 0
        for batch in eval_loader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attn_mask, labels = batch

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                logits_ = outputs['logits']
                loss = outputs['loss']

            logits += logits_.detach().cpu().numpy().tolist()
            label_ids += labels.reshape(-1).cpu().numpy().tolist()

            eval_loss += loss.mean().item()

        eval_loss = eval_loss / len(eval_loader)
        eval_accuracy, eval_precision, eval_recall = accuracy(logits, label_ids)

        result = {'eval_loss': eval_loss,
                    'eval_accuracy': eval_accuracy,
                    'eval_precision': eval_precision,
                    'eval_recall': eval_recall,
                    'train_loss': train_loss / step_per_epoch}

        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

        if eval_accuracy >= best_accuracy:
            torch.save(model.state_dict(), "model/model_best_back.pt")
            best_accuracy = eval_accuracy

if __name__ == '__main__':
    main()