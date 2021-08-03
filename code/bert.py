import torch
import torch.nn as nn
import os
import argparse
import json
from tqdm import tqdm
import numpy as np
import logging
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from transformers.models.deit.configuration_deit import DeiTConfig
from makedata import getTrainData,getEvalData
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from torch.utils.data.distributed import DistributedSampler
import pickle

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def accuracy(logits,label):
    outputs = np.argmax(logits, axis=1)
    return np.sum(outputs == label)

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_type",
                        default='base',
                        type=str)
    parser.add_argument("--gpu_ids",
                        default='0,1,2,3,4,5,6,7',
                        type=str)
    parser.add_argument("--task_name",
                        default='bert',
                        type=str)
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--a_dropout_prob",
                        default=0.1,
                        type=float)
    parser.add_argument("--h_dropout_prob",
                        default=0.1,
                        type=float)
    parser.add_argument("--s_dropout_prob",
                        default=0.1,
                        type=float)
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=8,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device %s n_gpu %d distributed training %r",
                device, n_gpu, bool(args.local_rank != -1))

    pretrained="bert-base-chinese"

    model_config = BertConfig.from_pretrained(
        pretrained, attention_probs_dropout_prob=args.a_dropout_prob, hidden_dropout_prob=args.h_dropout_prob,
        summary_last_dropout=args.s_dropout_prob)

    model = BertForSequenceClassification.from_pretrained(
        pretrained, config=model_config)
    tokenizer = BertTokenizer.from_pretrained(pretrained)

    torch.cuda.empty_cache()
    model.to(device)
    model = torch.nn.DataParallel(model)

    loss_fct = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.learning_rate)

    feature_file = "data/train_features.pkl"
    if os.path.exists(feature_file):
        train_dataset = pickle.load(open(feature_file, 'rb'))
    else:
        train_dataset = getTrainData(tokenizer)
        with open(feature_file, 'wb') as w:
            pickle.dump(train_dataset, w)

    feature_file = "data/eval_features.pkl"
    if os.path.exists(feature_file):
        eval_dataset = pickle.load(open(feature_file, 'rb'))
    else:
        eval_dataset = getEvalData(tokenizer)
        with open(feature_file, 'wb') as w:
            pickle.dump(eval_dataset, w)

    if args.local_rank == -1:
        sampler = SequentialSampler(train_dataset)
    else:
        sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, sampler=sampler, batch_size=args.train_batch_size, drop_last=False)
    eval_loader = DataLoader(
        eval_dataset, sampler=sampler, batch_size=args.eval_batch_size, drop_last=False)

    step_per_epoch=len(train_loader)
    best_accuracy = 0
    total_loss = 0

    logger.info("***** Running training *****")
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", step_per_epoch)

    for ie in range(int(args.num_train_epochs)):
        model.train()
        with tqdm(total=step_per_epoch, desc='Epoch %d' % (ie + 1)) as pbar:
            for batch in train_loader:
                batch = tuple(t.to(device) for t in batch)
                input_ids, attn_mask, labels = batch

                loss = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)[0]
                if n_gpu > 1:
                    loss = loss.mean()  
                total_loss+=loss.item()

                loss.backward()
                optimizer.step()  # We have accumulated enough gradients
                model.zero_grad()
                pbar.set_postfix(
                    {'loss': "%.3f"%loss})
                pbar.update(1)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        for batch in tqdm(eval_loader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, attn_mask, labels = batch

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                logits = outputs['logits']
                loss = outputs['loss']

            logits = logits.detach().cpu().numpy()
            label_ids = labels.cpu().numpy()

            tmp_eval_accuracy = accuracy(logits, label_ids.reshape(-1))

            eval_loss += loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

        result = {'eval_loss': eval_loss,
                    'eval_accuracy': eval_accuracy,
                    'loss': total_loss / step_per_epoch}

        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

        if eval_accuracy >= best_accuracy:
            torch.save(model.state_dict(), "model/model_best.pt")
            best_accuracy = eval_accuracy

if __name__ == '__main__':
    main()