# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and 
# The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
"""BERT finetuning runner with transformers."""

from __future__ import absolute_import, division, print_function
from collections import OrderedDict
import argparse
import csv
import logging
import os
import random
import sys
import io
import json

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from transformers import (BertConfig,
                          BertTokenizer,
                          BertForSequenceClassification,
                          AdamW,
                          get_linear_schedule_with_warmup,
                          WEIGHTS_NAME,
                          CONFIG_NAME)

# You can still use tensorboardX for logging if you wish:
from tensorboardX import SummaryWriter
from pprint import pprint

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs an InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. Untokenized text of the first sequence.
            text_b: (Optional) string. Untokenized text of the second sequence.
            label: (Optional) string. The label of the example.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification datasets."""

    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            idx = 0
            for line in reader:
                idx += 1
                # if idx > 100: break  # For quick debug, if needed
                lines.append(line)
            return lines


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir, dataset="dev"):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "{}.tsv".format(dataset))), dataset)

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                # column_types = [int(x) for x in line[2].split()]
                column_types = line[2].split()
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append((InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label),
                             column_types))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode, fact_place=None, balance=False, verbose=False):
    """Loads a data file into a list of `InputFeatures`."""
    assert fact_place is not None
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    pos_buf = []
    neg_buf = []
    logger.info("convert_examples_to_features ...")
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        example, column_types = example
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Truncate sequence pair so total length <= max_seq_length - 3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # For single sequence, account for [CLS] and [SEP].
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # If fact is "first", tokens_b is at the front; if "second", tokens_b is at the end
        if fact_place == "first":
            tokens = ["[CLS]"] + tokens_b + ["[SEP]"]
            segment_ids = [0] * (len(tokens_b) + 2)
            tokens += tokens_a + ["[SEP]"]
            segment_ids += [1] * (len(tokens_a) + 1)
        else:
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * (len(tokens_a) + 2)
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if verbose and ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        if balance:
            if label_id == 1:
                pos_buf.append(InputFeatures(input_ids, input_mask, segment_ids, label_id))
            else:
                neg_buf.append(InputFeatures(input_ids, input_mask, segment_ids, label_id))

            if len(pos_buf) > 0 and len(neg_buf) > 0:
                features.append(pos_buf.pop(0))
                features.append(neg_buf.pop(0))
        else:
            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "qqp":
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)


def main():
    logger.info("Running %s" % ' '.join(sys.argv))

    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--scan",
                        default="horizontal",
                        choices=["vertical", "horizontal"],
                        type=str,
                        help="The direction of linearizing table cells.")
    parser.add_argument("--data_dir",
                        default="../processed_datasets",
                        type=str,
                        help="The input data dir. Should contain .tsv files for the task.")
    parser.add_argument("--output_dir",
                        default="outputs",
                        type=str,
                        help="The output directory where model checkpoints will be written.")
    parser.add_argument("--load_dir",
                        type=str,
                        help="Directory where the model checkpoints will be loaded during evaluation")
    parser.add_argument('--load_step',
                        type=int,
                        default=0,
                        help="Checkpoint step to be loaded")
    parser.add_argument("--fact",
                        default="first",
                        choices=["first", "second"],
                        type=str,
                        help="Whether to put fact in front.")
    parser.add_argument("--test_set",
                        default="dev",
                        choices=["dev", "test", "simple_test", "complex_test", "small_test"],
                        help="Which test set is used for evaluation",
                        type=str)
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Batch size for eval.")
    parser.add_argument("--balance",
                        action='store_true',
                        help="Balance between + and - samples for training.")
    ## Other parameters
    parser.add_argument("--bert_model",
                        default="bert-base-multilingual-cased",
                        type=str,
                        help="Bert pre-trained model in the list: "
                             "bert-base-uncased, bert-large-uncased, bert-base-cased, "
                             "bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default="QQP",
                        type=str,
                        help="The name of the task to train.")
    parser.add_argument('--period',
                        type=int,
                        default=500)
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="Maximum total input sequence length after tokenization.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=6,
                        type=int,
                        help="Batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=20.0,
                        type=float,
                        help="Total number of training epochs.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Linear learning rate warmup proportion of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on GPUs")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="Random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before a backward pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling for fp16 numeric stability.")
    parser.add_argument('--server_ip', type=str, default='', help="For remote debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For remote debugging.")
    args = parser.parse_args()
    pprint(vars(args))
    sys.stdout.flush()

    if args.server_ip and args.server_port:
        # Remote debugging
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {
        "qqp": QqpProcessor,
    }

    output_modes = {
        "qqp": "classification",
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()

        # Check if MPS is available and use if cuda is not available (for better performance on Apple)
        if not torch.cuda.is_available() and torch.backends.mps.is_available():
            device = torch.device("mps")
        
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient_accumulation_steps parameter must be >= 1")

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.output_dir = "{}_fact-{}_{}".format(args.output_dir, args.fact, args.scan)
    args.data_dir = os.path.join(args.data_dir, "tsv_data_{}".format(args.scan))
    logger.info("Datasets are loaded from {}\nOutputs will be saved to {}".format(args.data_dir, args.output_dir))
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    writer = SummaryWriter(os.path.join(args.output_dir, 'events'))

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load tokenizer from transformers
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps
        ) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Use cache_dir if provided, else None
    cache_dir = args.cache_dir if args.cache_dir else None

    # Load model from transformers
    if args.load_dir:
        load_dir = args.load_dir
    else:
        load_dir = args.bert_model

    model = BertForSequenceClassification.from_pretrained(
        load_dir,
        cache_dir=cache_dir,
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False
    )

    if args.fp16:
        model.half()

    model.to(device)
    if args.local_rank != -1:
        # For distributed training (and optional FP16 with apex)
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex "
                              "to use distributed and fp16 training.")
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.do_train:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex for FP16.")
            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

            # We create our own scheduler
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(args.warmup_proportion * num_train_optimization_steps),
                num_training_steps=num_train_optimization_steps
            )

        else:
            # Use AdamW + HF schedule
            optimizer = AdamW(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              correct_bias=False)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(args.warmup_proportion * num_train_optimization_steps),
                num_training_steps=num_train_optimization_steps
            )

    global_step = 0
    tr_loss = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode,
            fact_place=args.fact, balance=args.balance)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        else:  # regression
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            logger.info("Training epoch {} ...".format(epoch))
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                # Forward pass in transformers
                outputs = model(input_ids=input_ids,
                                token_type_ids=segment_ids,
                                attention_mask=input_mask)
                logits = outputs[0]  # model outputs are always tuples

                if output_mode == "classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                else:  # regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), label_ids.view(-1))

                if n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                writer.add_scalar('train/loss', loss, global_step)
                tr_loss += loss.item()

                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # Gradient clipping for normal (non-apex) usage
                    if not args.fp16:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    model.zero_grad()
                    global_step += 1

                if (step + 1) % args.period == 0:
                    # Save a trained model, configuration and tokenizer
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_dir = os.path.join(args.output_dir, 'save_step_{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    # Instead of manually saving state_dict, use save_pretrained
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    # Evaluate at this step
                    model.eval()
                    torch.set_grad_enabled(False)
                    evaluate(args, model, device, processor, label_list, num_labels, tokenizer,
                             output_mode, tr_loss, global_step, task_name, tbwriter=writer,
                             save_dir=output_dir)
                    model.train()
                    torch.set_grad_enabled(True)
                    tr_loss = 0

    # do eval before exit
    if args.do_eval:
        if not args.do_train:
            global_step = 0
            output_dir = None
        save_dir = output_dir if args.do_train else args.load_dir
        if not save_dir:
            save_dir = args.output_dir
        tbwriter = SummaryWriter(os.path.join(save_dir, 'eval/events'))
        load_step = args.load_step
        if args.load_dir is not None:
            try:
                load_step = int(os.path.split(args.load_dir)[1].replace('save_step_', ''))
                print("load_step = {}".format(load_step))
            except:
                pass
        model.eval()
        evaluate(args, model, device, processor, label_list, num_labels, tokenizer, output_mode,
                 tr_loss, global_step, task_name, tbwriter=tbwriter, save_dir=save_dir, load_step=load_step)


def evaluate(args, model, device, processor, label_list, num_labels, tokenizer, output_mode,
             tr_loss, global_step, task_name, tbwriter=None, save_dir=None, load_step=0):

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_dev_examples(args.data_dir, dataset=args.test_set)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer,
            output_mode, fact_place=args.fact, balance=False)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        else:
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        batch_idx = 0
        eval_loss = 0
        nb_eval_steps = 0
        preds = []
        temp = []

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids,
                                token_type_ids=segment_ids,
                                attention_mask=input_mask)
                logits = outputs[0]

            if output_mode == "classification":
                loss_fct = CrossEntropyLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            else:
                loss_fct = MSELoss()
                tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

            labels = label_ids.detach().cpu().numpy().tolist()
            start = batch_idx * args.eval_batch_size
            end = start + len(labels)
            batch_range = list(range(start, end))
            csv_names = [eval_examples[i][0].guid.replace("{}-".format(args.test_set), "") for i in batch_range]
            facts = [eval_examples[i][0].text_b for i in batch_range]
            # gold labels
            assert len(csv_names) == len(facts) == len(labels)

            temp.extend([(x, y, z) for x, y, z in zip(csv_names, facts, labels)])
            batch_idx += 1

        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]
        if output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        else:
            preds = np.squeeze(preds)

        evaluation_results = OrderedDict()
        for x, y in zip(temp, preds):
            c, f, l = x
            if c not in evaluation_results:
                evaluation_results[c] = [{'fact': f, 'gold': int(l), 'pred': int(y)}]
            else:
                evaluation_results[c].append({'fact': f, 'gold': int(l), 'pred': int(y)})

        logger.info("save_dir is {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        output_eval_file = os.path.join(save_dir, "{}_eval_results.json".format(args.test_set))
        with io.open(output_eval_file, "w", encoding='utf-8') as fout:
            json.dump(evaluation_results, fout, sort_keys=True, indent=4)

        result = compute_metrics(task_name, preds, all_label_ids.numpy())
        loss = tr_loss / args.period if args.do_train and global_step > 0 else None

        log_step = global_step if args.do_train and global_step > 0 else load_step
        result['eval_loss'] = eval_loss
        result['global_step'] = log_step
        result['loss'] = loss

        output_eval_metrics = os.path.join(save_dir, "eval_metrics.txt")
        with open(output_eval_metrics, "a") as writer:
            logger.info("***** Eval results {}*****".format(args.test_set))
            writer.write("***** Eval results {}*****\n".format(args.test_set))
            for key in sorted(result.keys()):
                if result[key] is not None and tbwriter is not None:
                    tbwriter.add_scalar('{}/{}'.format(args.test_set, key), result[key], log_step)
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    main()