# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors,
# The HuggingFace Inc. team, NVIDIA CORPORATION, and others.
# Licensed under the Apache License, Version 2.0.

"""
BERT finetuning runner with transformers, refactored to accept a dictionary of args
instead of command-line flags, allowing function-based usage (e.g. from a Jupyter notebook).
"""

import os
import sys
import io
import csv
import json
import argparse
import random
import logging
import numpy as np
import torch

from tqdm import tqdm, trange
from pprint import pprint
from collections import OrderedDict

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import (
    BertConfig,
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
    WEIGHTS_NAME,
    CONFIG_NAME
)

from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Data structures
# -------------------------------------------------------------------------
class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, text_b=None, label=None):
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
        lines = []
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            for line in reader:
                lines.append(line)
        return lines

class QqpProcessor(DataProcessor):
    """Processor for the QQP data set."""
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
                column_types = line[2].split()
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            examples.append((example, column_types))
        return examples


# -------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------
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

def f1_for_binary_classification(preds, labels):
    return f1_score(y_true=labels, y_pred=preds)

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1_val = f1_for_binary_classification(preds, labels)
    return {
        "acc": acc,
        "f1": f1_val,
        "acc_and_f1": (acc + f1_val) / 2,
    }

def compute_metrics(task_name, preds, labels):
    if task_name == "qqp":
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode, fact_place=None,
                                 balance=False, verbose=False):
    """
    Loads a data file into a list of `InputFeatures`.
    """
    assert fact_place is not None

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    pos_buf = []
    neg_buf = []

    for (ex_index, example) in enumerate(examples):
        example, column_types = example

        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Truncate sequence pair so total length <= max_seq_length - 3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # For single sequence
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # Fact placement logic
        if fact_place == "first":
            tokens = ["[CLS]"] + (tokens_b or []) + ["[SEP]"]
            segment_ids = [0] * (len(tokens_b or []) + 2)
            tokens += tokens_a + ["[SEP]"]
            segment_ids += [1] * (len(tokens_a) + 1)
        else:
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * (len(tokens_a) + 2)
            tokens += (tokens_b or []) + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b or []) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-padding
        padding_len = max_seq_length - len(input_ids)
        input_ids += [0] * padding_len
        input_mask += [0] * padding_len
        segment_ids += [0] * padding_len

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        else:  # regression
            label_id = float(example.label)

        # Balance or not
        if balance:
            if label_id == 1:
                pos_buf.append(InputFeatures(input_ids, input_mask, segment_ids, label_id))
            else:
                neg_buf.append(InputFeatures(input_ids, input_mask, segment_ids, label_id))
            if len(pos_buf) > 0 and len(neg_buf) > 0:
                features.append(pos_buf.pop(0))
                features.append(neg_buf.pop(0))
        else:
            features.append(InputFeatures(input_ids, input_mask, segment_ids, label_id))

    return features


# -------------------------------------------------------------------------
# Main evaluation routine
# -------------------------------------------------------------------------
def evaluate(args, model, device, processor, label_list, num_labels, tokenizer,
             output_mode, tr_loss, global_step, tbwriter=None, save_dir=None, load_step=0):
    """
    Evaluates on the dev/test set. Returns relevant metrics, plus
    predictions and labels (and, optionally, predicted probabilities for binary classification).
    """
    # Load the evaluation data
    eval_examples = processor.get_dev_examples(args['data_dir'], dataset=args['test_set'])
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args['max_seq_length'], tokenizer,
        output_mode, fact_place=args['fact'], balance=False
    )

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args['eval_batch_size'])

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    else:
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args['eval_batch_size'])

    batch_idx = 0
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    all_labels = []
    all_pred_probs = []  # for ROC/PR curves if binary classification

    temp = []  # storing (example_id, fact, label) for JSON output

    model.eval()
    torch.set_grad_enabled(False)

    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        outputs = model(
            input_ids=input_ids,
            token_type_ids=segment_ids,
            attention_mask=input_mask
        )
        logits = outputs[0]  # model outputs are always tuples

        # Compute eval loss
        if output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        else:
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

        # Predictions
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

        # Save labels for later
        all_labels.extend(label_ids.cpu().numpy().tolist())

        # If binary classification, also store predicted probability of label 1:
        if output_mode == "classification" and num_labels == 2:
            # Softmax along axis=1
            softmax_vals = torch.softmax(logits, dim=1)
            # Probability that label=1
            prob_of_one = softmax_vals[:, 1].detach().cpu().numpy().tolist()
            all_pred_probs.extend(prob_of_one)
        else:
            # For multi-class or regression, can store all class probabilities or raw logits
            # Here, let's store raw logits for reference
            all_pred_probs.extend(logits.detach().cpu().numpy().tolist())

        # For writing JSON results
        labels_batch = label_ids.detach().cpu().numpy().tolist()
        start = batch_idx * args['eval_batch_size']
        end = start + len(labels_batch)
        batch_range = list(range(start, end))

        # For each entry, we have: (example, column_types)
        # So eval_examples[i][0] is the InputExample
        # We'll store the "guid" (unique ID), the "fact" (text_b), and gold label
        for i_idx, lab in zip(batch_range, labels_batch):
            guid = eval_examples[i_idx][0].guid.replace("{}-".format(args['test_set']), "")
            fact = eval_examples[i_idx][0].text_b
            temp.append((guid, fact, int(lab)))

        batch_idx += 1

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]

    # If classification, pick the argmax
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    else:
        preds = np.squeeze(preds)

    # Build evaluation results for JSON
    evaluation_results = OrderedDict()
    for x, y in zip(temp, preds):
        c, f, l = x
        if c not in evaluation_results:
            evaluation_results[c] = [{'fact': f, 'gold': int(l), 'pred': int(y)}]
        else:
            evaluation_results[c].append({'fact': f, 'gold': int(l), 'pred': int(y)})

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    output_eval_file = os.path.join(save_dir, "{}_eval_results.json".format(args['test_set']))
    with io.open(output_eval_file, "w", encoding='utf-8') as fout:
        json.dump(evaluation_results, fout, sort_keys=True, indent=4)

    # Compute final metrics
    labels_array = np.array(all_labels)
    if output_mode == "classification":
        results_dict = compute_metrics(args['task_name'].lower(), preds, labels_array)
    else:
        # For regression or other tasks, you could define your own metrics
        results_dict = {"mse": float(np.mean((preds - labels_array) ** 2))}

    eval_loss_val = float(eval_loss)
    results_dict['eval_loss'] = eval_loss_val

    # Save raw metrics
    output_eval_metrics = os.path.join(save_dir, "eval_metrics.txt")
    with open(output_eval_metrics, "a") as writer:
        writer.write("***** Eval results {} *****\n".format(args['test_set']))
        for key in sorted(results_dict.keys()):
            writer.write("%s = %s\n" % (key, str(results_dict[key])))

    # Possibly log to tensorboard
    log_step = global_step if args.get('do_train', False) and global_step > 0 else load_step
    if tbwriter is not None:
        for key in results_dict:
            tbwriter.add_scalar('{}/{}'.format(args['test_set'], key), results_dict[key], log_step)

    # Return relevant info for plotting
    # For binary classification, all_pred_probs are the probabilities of label=1
    return {
        "eval_results": results_dict,
        "labels": all_labels,
        "preds": preds,
        "probs": all_pred_probs
    }


# -------------------------------------------------------------------------
# The main function-based entry point
# -------------------------------------------------------------------------
def run_bert_experiment(args_dict):
    """
    Replicates the logic of run_BERT.py, but uses a dictionary of args instead
    of command-line arguments, and returns relevant metrics for further analysis.
    
    Example usage from a notebook:
        args = {
            'do_train': False,
            'do_eval': True,
            'scan': 'horizontal',
            'fact': 'first',
            'load_dir': 'outputs_fact-first_horizontal_snapshot/save_step_12500',
            'eval_batch_size': 16,
            # etc...
        }
        results = run_bert_experiment(args)
    """
    # Convert the user-provided dictionary into local args references
    args = args_dict.copy()

    # Logger config
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )
    logger.info("Running BERT with provided dictionary arguments:")
    pprint(args)

    # Device setup
    if args.get('no_cuda', False):
        device = torch.device("cpu")
        n_gpu = 0
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            n_gpu = torch.cuda.device_count()
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            n_gpu = 1
        else:
            device = torch.device("cpu")
            n_gpu = 0

    logger.info(f"Device: {device}, n_gpu: {n_gpu}")

    # Must do either train or eval
    if not args.get('do_train', False) and not args.get('do_eval', False):
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # Some defaults, in case not specified
    args.setdefault('task_name', 'qqp')
    args.setdefault('period', 500)
    args.setdefault('num_train_epochs', 20.0)
    args.setdefault('gradient_accumulation_steps', 1)
    args.setdefault('seed', 42)
    args.setdefault('train_batch_size', 6)
    args.setdefault('eval_batch_size', 16)
    args.setdefault('learning_rate', 5e-5)
    args.setdefault('max_seq_length', 512)
    args.setdefault('warmup_proportion', 0.1)
    args.setdefault('bert_model', 'bert-base-multilingual-cased')
    args.setdefault('do_lower_case', False)
    args.setdefault('local_rank', -1)
    args.setdefault('fp16', False)
    args.setdefault('loss_scale', 0)
    args.setdefault('test_set', 'dev')
    args.setdefault('balance', False)

    # Adjust output dir if needed
    if 'output_dir' not in args:
        args['output_dir'] = "outputs"
    args['output_dir'] = "{}_fact-{}_{}".format(args['output_dir'], args['fact'], args['scan'])

    # Adjust data_dir
    if 'data_dir' not in args:
        args['data_dir'] = "../processed_datasets"
    args['data_dir'] = os.path.join(args['data_dir'], "tsv_data_{}".format(args['scan']))

    logger.info("Datasets are loaded from {}\nOutputs will be saved to {}".format(args['data_dir'], args['output_dir']))

    # Make sure output_dir exists
    if not os.path.exists(args['output_dir']):
        os.makedirs(args['output_dir'])

    # Tensorboard writer
    writer = SummaryWriter(os.path.join(args['output_dir'], 'events'))

    # Task
    task_name = args['task_name'].lower()
    processors = {"qqp": QqpProcessor}
    output_modes = {"qqp": "classification"}

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Initialize random seeds
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args['seed'])

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.get('bert_model'), do_lower_case=args['do_lower_case'])

    # Possibly prepare training data
    train_examples = None
    num_train_optimization_steps = None
    if args.get('do_train', False):
        train_examples = processor.get_train_examples(args['data_dir'])
        num_train_optimization_steps = int(
            len(train_examples) / args['train_batch_size'] / args['gradient_accumulation_steps']
        ) * args['num_train_epochs']
        logger.info("Total optimization steps: %d", num_train_optimization_steps)
    else:
        num_train_optimization_steps = 0

    # Load model
    if 'load_dir' in args and args['load_dir'] is not None:
        load_dir = args['load_dir']
    else:
        load_dir = args['bert_model']

    model = BertForSequenceClassification.from_pretrained(
        load_dir,
        cache_dir=args.get('cache_dir', None),
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False
    )

    if args['fp16']:
        model.half()

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.get('do_train', False):
        train_features = convert_examples_to_features(
            train_examples, label_list, args['max_seq_length'], tokenizer,
            output_mode, fact_place=args['fact'], balance=args['balance']
        )
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        else:
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args['train_batch_size'])

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

        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args['learning_rate'],
                          correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args['warmup_proportion'] * num_train_optimization_steps),
            num_training_steps=num_train_optimization_steps
        )
    else:
        train_dataloader = None
        optimizer = None
        scheduler = None

    global_step = 0
    tr_loss = 0.0

    # -----------------------------------------
    # Training loop
    # -----------------------------------------
    if args.get('do_train', False) and train_dataloader is not None:
        model.train()
        for epoch in trange(int(args['num_train_epochs']), desc="Epoch"):
            logger.info("Training epoch {} ...".format(epoch))
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                outputs = model(
                    input_ids=input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask
                )
                logits = outputs[0]

                if output_mode == "classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                else:
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), label_ids.view(-1))

                if n_gpu > 1:
                    loss = loss.mean()
                if args['gradient_accumulation_steps'] > 1:
                    loss = loss / args['gradient_accumulation_steps']

                loss.backward()
                writer.add_scalar('train/loss', loss.item(), global_step)
                tr_loss += loss.item()

                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                # Gradient update
                if (step + 1) % args['gradient_accumulation_steps'] == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    model.zero_grad()
                    global_step += 1

                # Periodic saving & evaluation
                if (step + 1) % args['period'] == 0:
                    model.eval()
                    torch.set_grad_enabled(False)

                    save_subdir = os.path.join(args['output_dir'], f'save_step_{global_step}')
                    if not os.path.exists(save_subdir):
                        os.makedirs(save_subdir)

                    # Save model/ tokenizer
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(save_subdir)
                    tokenizer.save_pretrained(save_subdir)

                    # Evaluate
                    evaluate(
                        args, model, device, processor, label_list, num_labels, tokenizer,
                        output_mode, tr_loss, global_step, tbwriter=writer, save_dir=save_subdir, load_step=global_step
                    )

                    model.train()
                    torch.set_grad_enabled(True)
                    tr_loss = 0.0

    # -----------------------------------------
    # Final evaluation (if do_eval)
    # -----------------------------------------
    result_dict = None
    if args.get('do_eval', False):
        # If not training, we might have load_dir
        if not args.get('do_train', False):
            global_step = 0
        save_dir = args.get('load_dir', None) or args['output_dir']

        # Evaluate
        result_dict = evaluate(
            args, model, device, processor, label_list, num_labels, tokenizer,
            output_mode, tr_loss, global_step, tbwriter=writer, save_dir=save_dir, load_step=global_step
        )
    else:
        logger.info("No eval performed because do_eval=False")

    writer.close()
    return result_dict