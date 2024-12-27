import json
import os
import sys
import time
import random
import torch
import numpy
from collections import Counter
from itertools import chain
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn

from PRA_data import get_batch
from Transformer import Encoder, Decoder


def back_to_words(seq, ivocab):
    """
    Convert a sequence of token IDs back to words, omitting <PAD>.
    """
    return " ".join([ivocab[_id.item()] for _id in seq if ivocab[_id.item()] != "<PAD>"])


def evaluate(val_dataloader, encoder_stat, encoder_prog, args, device, ivocab):
    """
    Evaluation logic exactly as in the original code, but now also collects
    all predicted probabilities and ground-truth labels, so we can plot
    confusion matrices and ROC curves later.
    """
    mapping = {}
    TP, TN, FN, FP = 0, 0, 0, 0

    # These lists will store all ground-truth labels and predicted probabilities
    # for the entire dataset:
    all_labels = []
    all_probs = []

    # Put model(s) in eval mode.
    encoder_stat.eval()
    encoder_prog.eval()

    for val_step, batch in enumerate(val_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, prog_ids, labels, index, true_lab, pred_lab = batch

        enc_stat = encoder_stat(input_ids)
        enc_prog, logits = encoder_prog(prog_ids, input_ids, enc_stat)

        # Predicted probability for each example
        similarity = torch.sigmoid(logits)        # shape: (batch_size,)
        similarity_np = similarity.cpu().data.numpy()

        # Binarize predictions based on threshold
        sim = (similarity_np > args['threshold']).astype('float32')

        labels_np = labels.cpu().data.numpy()
        index_np = index.cpu().data.numpy()
        true_lab_np = true_lab.cpu().data.numpy()
        pred_lab_np = pred_lab.cpu().data.numpy()

        # Collect for ROC, confusion matrix, etc.
        all_labels.extend(labels_np.tolist())
        all_probs.extend(similarity_np.tolist())

        # Compute the confusion matrix values
        TP += ((sim == 1) & (labels_np == 1)).sum()
        TN += ((sim == 0) & (labels_np == 0)).sum()
        FN += ((sim == 0) & (labels_np == 1)).sum()
        FP += ((sim == 1) & (labels_np == 0)).sum()

        # "mapping" logic for final accuracy
        if not args['voting']:
            # Non-voting logic
            for i, s, p, t, inp_id, prog_id in zip(index_np, similarity_np, pred_lab_np, true_lab_np, input_ids, prog_ids):
                if args['analyze']:
                    inp_str = back_to_words(inp_id, ivocab)
                    prog_str = back_to_words(prog_id[1:], ivocab)
                else:
                    inp_str = None
                    prog_str = None

                if i not in mapping:
                    mapping[i] = [s, p.item(), t.item(), inp_str, prog_str]
                else:
                    if s > mapping[i][0]:
                        mapping[i] = [s, p.item(), t.item(), inp_str, prog_str]
        else:
            # Voting logic
            factor = 2
            for i, s, p, t in zip(index_np, similarity_np, pred_lab_np, true_lab_np):
                if i not in mapping:
                    if p == 1:
                        mapping[i] = [factor * s, s, t]
                    else:
                        mapping[i] = [-s, s, t]
                else:
                    if p == 1:
                        mapping[i][0] += factor * s
                    else:
                        mapping[i][0] -= s

    # Compute precision and recall
    precision = TP / (TP + FP + 0.001)
    recall = TP / (TP + FN + 0.001)

    print("TP: {}, FP: {}, FN: {}, TN: {}. precision = {}: recall = {}".format(
        TP, FP, FN, TN, precision, recall
    ))

    # For final accuracy, we consult the mapping dictionary
    accuracy = 0
    results = []
    if not args['voting']:
        success, fail = 0, 0
        for i, line in mapping.items():
            if line[1] == line[2]:
                success += 1
            else:
                fail += 1
            results.append({
                'pred': line[1],
                'gold': line[2],
                'fact': line[3],
                'program': line[4]
            })
        accuracy = success / (success + fail + 0.001)
        print("success = {}, fail = {}, accuracy = {}".format(success, fail, accuracy))
    else:
        success, fail = 0, 0
        for i, ent in mapping.items():
            # ent[0] is the "score" after voting, ent[2] is the gold label
            if (ent[0] > 0 and ent[2] == 1) or (ent[0] < 0 and ent[2] == 0):
                success += 1
            else:
                fail += 1
        accuracy = success / (success + fail + 0.001)
        print("success = {}, fail = {}, accuracy = {}".format(success, fail, accuracy))

    # If analyze mode is on, we save results
    if args['analyze']:
        if args['do_test'] or args['do_small_test']:
            with open('/tmp/test_eval_results.json', 'w') as f:
                json.dump(results, f, indent=2)
        if args['do_val']:
            with open('/tmp/val_eval_results.json', 'w') as f:
                json.dump(results, f, indent=2)

    return precision, recall, accuracy, all_labels, all_probs


def run_experiment(args):
    """
    This function encapsulates the original script's logic so that it can be
    imported and run from within a notebook or elsewhere. Pass in an args dict
    containing the flags and hyperparameters. For example:

    args = {
        'do_train': True,
        'do_val': True,
        'do_test': False,
        'do_simple_test': False,
        'do_complex_test': False,
        'do_small_test': False,
        'emb_dim': 128,
        'dropout': 0.2,
        'resume': False,
        'batch_size': 512,
        'data_dir': '../preprocessed_data_program/',
        'max_seq_length': 100,
        'layer_num': 3,
        'voting': False,
        'id': "0",
        'analyze': False,
        'threshold': 0.5,
        'output_dir': 'checkpoints/',
        'learning_rate': 5e-4
    }

    Then call: run_experiment(args).
    """

    # -------------------------------------------------------------------------
    # 1. Setup the device
    # -------------------------------------------------------------------------
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')

    # -------------------------------------------------------------------------
    # 2. Make sure output directory exists
    # -------------------------------------------------------------------------
    if not os.path.exists(args['output_dir']):
        os.mkdir(args['output_dir'])

    # -------------------------------------------------------------------------
    # 3. Load vocabulary
    # -------------------------------------------------------------------------
    with open(os.path.join(args['data_dir'], 'vocab.json')) as f:
        vocab = json.load(f)
    ivocab = {w: k for k, w in vocab.items()}

    # -------------------------------------------------------------------------
    # 4. Optionally prepare DataLoaders for train/val/test
    # -------------------------------------------------------------------------
    start_time = time.time()
    
    train_dataloader = None
    val_dataloader = None

    # If do_train is True, prepare train loader
    if args.get('do_train', False):
        train_examples = get_batch(
            option='train',
            data_dir=args['data_dir'],
            vocab=vocab,
            max_seq_length=args['max_seq_length'],
            cutoff=-1
        )
        train_data = TensorDataset(*train_examples)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args['batch_size'])

    # If do_val is True, prepare validation loader
    if args.get('do_val', False):
        val_examples = get_batch(
            option='val',
            data_dir=args['data_dir'],
            vocab=vocab,
            max_seq_length=args['max_seq_length']
        )
        val_data = TensorDataset(*val_examples)
        val_sampler = SequentialSampler(val_data)
        val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=args['batch_size'])

    # If do_test is True
    if args.get('do_test', False):
        test_examples = get_batch(
            option='test',
            data_dir=args['data_dir'],
            vocab=vocab,
            max_seq_length=args['max_seq_length']
        )
        test_data = TensorDataset(*test_examples)
        test_sampler = SequentialSampler(test_data)
        val_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args['batch_size'])

    # If do_simple_test is True
    if args.get('do_simple_test', False):
        simple_examples = get_batch(
            option='simple_test',
            data_dir=args['data_dir'],
            vocab=vocab,
            max_seq_length=args['max_seq_length']
        )
        simple_data = TensorDataset(*simple_examples)
        simple_sampler = SequentialSampler(simple_data)
        val_dataloader = DataLoader(simple_data, sampler=simple_sampler, batch_size=args['batch_size'])

    # If do_complex_test is True
    if args.get('do_complex_test', False):
        complex_examples = get_batch(
            option='complex_test',
            data_dir=args['data_dir'],
            vocab=vocab,
            max_seq_length=args['max_seq_length']
        )
        complex_data = TensorDataset(*complex_examples)
        complex_sampler = SequentialSampler(complex_data)
        val_dataloader = DataLoader(complex_data, sampler=complex_sampler, batch_size=args['batch_size'])

    # If do_small_test is True
    if args.get('do_small_test', False):
        small_examples = get_batch(
            option='small_test',
            data_dir=args['data_dir'],
            vocab=vocab,
            max_seq_length=args['max_seq_length']
        )
        small_data = TensorDataset(*small_examples)
        small_sampler = SequentialSampler(small_data)
        val_dataloader = DataLoader(small_data, sampler=small_sampler, batch_size=args['batch_size'])

    print("Loading used {} secs".format(time.time() - start_time))

    # -------------------------------------------------------------------------
    # 5. Initialize models (Encoder + Decoder)
    # -------------------------------------------------------------------------
    encoder_stat = Encoder(
        vocab_size=len(vocab),
        d_word_vec=128,
        n_layers=args['layer_num'],
        d_model=128,
        n_head=4
    )
    encoder_prog = Decoder(
        vocab_size=len(vocab),
        d_word_vec=128,
        n_layers=args['layer_num'],
        d_model=128,
        n_head=4
    )

    encoder_stat.to(device)
    encoder_prog.to(device)

    # -------------------------------------------------------------------------
    # 6. If resume is True, load saved weights
    # -------------------------------------------------------------------------
    if args.get('resume', False):
        stat_path = os.path.join(args['output_dir'], f"encoder_stat_{args['id']}.pt")
        prog_path = os.path.join(args['output_dir'], f"encoder_prog_{args['id']}.pt")
        encoder_stat.load_state_dict(torch.load(stat_path, map_location=device))
        encoder_prog.load_state_dict(torch.load(prog_path, map_location=device))
        print("Reloading saved model from", args['output_dir'])

    # -------------------------------------------------------------------------
    # 7. Training loop (if do_train=True)
    # -------------------------------------------------------------------------
    if args.get('do_train', False) and train_dataloader is not None:
        loss_func = nn.BCEWithLogitsLoss(reduction="mean").to(device)

        encoder_stat.train()
        encoder_prog.train()

        print("Start Training with {} batches".format(len(train_dataloader)))

        # Combine parameters
        params = chain(encoder_stat.parameters(), encoder_prog.parameters())
        optimizer = torch.optim.Adam(
            filter(lambda x: x.requires_grad, params),
            lr=args['learning_rate'],
            betas=(0.9, 0.98),
            eps=0.9e-09
        )

        best_accuracy = 0
        # The original code has a fixed 10 epochs
        for epoch in range(10):
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, prog_ids, labels, index, true_lab, pred_lab = batch

                encoder_stat.zero_grad()
                encoder_prog.zero_grad()
                optimizer.zero_grad()

                enc_stat = encoder_stat(input_ids)
                enc_prog, logits = encoder_prog(prog_ids, input_ids, enc_stat)

                loss = loss_func(logits, labels)
                loss.backward()
                optimizer.step()

                # Print loss every 20 steps
                if (step + 1) % 20 == 0:
                    similarity = torch.sigmoid(logits)
                    pred = (similarity > args['threshold']).float()
                    print(
                        "Epoch {}, Step {}: Loss = {:.4f}".format(epoch + 1, step + 1, loss.item()),
                        "\nPred:", list(pred[:10].cpu().data.numpy()),
                        "\nGold:", list(labels[:10].cpu().data.numpy())
                    )

                # Evaluate on val set every 200 steps
                if (step + 1) % 200 == 0 and args.get('do_val', False) and val_dataloader is not None:
                    encoder_stat.eval()
                    encoder_prog.eval()

                    precision, recall, accuracy = evaluate(val_dataloader, encoder_stat, encoder_prog, args, device, ivocab)

                    if accuracy > best_accuracy:
                        torch.save(
                            encoder_stat.state_dict(),
                            os.path.join(args['output_dir'], f"encoder_stat_{args['id']}.pt")
                        )
                        torch.save(
                            encoder_prog.state_dict(),
                            os.path.join(args['output_dir'], f"encoder_prog_{args['id']}.pt")
                        )
                        best_accuracy = accuracy

                    encoder_stat.train()
                    encoder_prog.train()

    # -------------------------------------------------------------------------
    # 8. Final evaluation if requested
    # -------------------------------------------------------------------------
    # If any of do_val / do_test / do_simple_test / do_complex_test / do_small_test is True,
    # we run evaluation. (The user’s original code does exactly this at the bottom.)
    if (args.get('do_val', False) or
        args.get('do_test', False) or
        args.get('do_simple_test', False) or
        args.get('do_complex_test', False) or
        args.get('do_small_test', False)) and val_dataloader is not None:

        encoder_stat.eval()
        encoder_prog.eval()
        precision, recall, accuracy, all_labels, all_probs = evaluate(
            val_dataloader, encoder_stat, encoder_prog, args, device, ivocab
        )
        
        # Return or store the results. E.g., we can return them:
        return {
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'labels': all_labels,
            'probs': all_probs
        }

    # If we get here without returning, let's return None or an empty dict
    return None