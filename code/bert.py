import torch
import torch.nn as nn
import os
import argparse
from tqdm import tqdm
import logging
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from makedata import getTrainData
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from torch.utils.data.distributed import DistributedSampler
import pickle

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()

git config --global user.name "awdrgyjilplij"
git config --global user.email "1979886892@qq.com"

    # Required parameters
    parser.add_argument("--model_type",
                        default='base',
                        type=str)
    parser.add_argument("--gpu_ids",
                        default='6,7',
                        type=str)
    parser.add_argument("--task_name",
                        default='bert',
                        type=str)
    parser.add_argument("--do_train",
                        default=True,
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=True,
                        help="Whether to run eval on the dev set.")
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
    parser.add_argument("--schedule",
                        default='warmup_linear',
                        type=str,
                        help='schedule')
    parser.add_argument("--weight_decay_rate",
                        default=0.01,
                        type=float,
                        help='weight_decay_rate')
    parser.add_argument("--num_train_epochs",
                        default=8,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--setting_file', type=str, default='setting.txt')
    parser.add_argument('--log_file', type=str, default='log.txt')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
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

    feature_file = "../data/train_features.pkl"
    if os.path.exists(feature_file):
        train_dataset = pickle.load(open(feature_file, 'rb'))
    else:
        train_dataset = getTrainData(tokenizer)
        with open(feature_file, 'wb') as w:
            pickle.dump(train_dataset, w)

    if args.local_rank == -1:
        sampler = SequentialSampler(train_dataset)
    else:
        sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, sampler=sampler, batch_size=args.eval_batch_size, drop_last=False)

    print(args.do_train)
    if args.do_train:
        print("train")
        best_accuracy = 0

        logger.info("***** Running training *****")
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", len(train_loader))

        for ie in range(int(args.num_train_epochs)):
            model.train()
            with tqdm(total=len(train_loader), desc='Epoch %d' % (ie + 1)) as pbar:
                for step, batch in enumerate(train_loader):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, label_ids = batch

                    loss = model(input_ids=input_ids,
                                 attention_mask=input_mask, labels=label_ids)[0]
                    if n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    loss.backward()

                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        optimizer.step()  # We have accumulated enough gradients
                        model.zero_grad()
                        pbar.set_postfix(
                            {'loss': '{0:1.5f}'.format(loss)})
                        pbar.update(1)

if __name__ == '__main__':
    main()