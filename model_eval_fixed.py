import json
from collections import Counter
import random
from PRA_data import get_batch
# from Transformer import Encoder, Decoder
from my_transformer import MultiLayerTransformer as Encoder
import argparse
import os
import sys
from itertools import chain
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import time
import torch.nn as nn
import torch.nn.functional as F
import numpy
import yaml



config = yaml.safe_load(open("config.yaml", "r"))
max_seq_length = config["max_seq_length"]
layer_num = config["layer_num"]
num_epoch = config["num_epoch"]
batch_size = config["batch_size"]
learning_rate = config["learning_rate"]
dropout = config["dropout"]
emb_dim = config["emb_dim"]
n_heads = config["n_heads"]
device = torch.device('cuda')

cutoff = 200
cutoff_val = 2



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_val', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_test', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_simple_test', default=False, action="store_true",
                        help="whether to train or test the model")
    parser.add_argument('--do_complex_test', default=False, action="store_true",
                        help="whether to train or test the model")
    parser.add_argument('--do_small_test', default=False, action="store_true",
                        help="whether to train or test the model")
    # parser.add_argument('--emb_dim', type=int, default=128, help="the embedding dimension")
    # parser.add_argument('--dropout', type=float, default=0.2, help="the embedding dimension")
    parser.add_argument('--resume', action='store_true', default=False, help="whether to resume previous run")
    # parser.add_argument('--batch_size', type=int, default=128, help="the embedding dimension")
    parser.add_argument('--data_dir', type=str, default='../preprocessed_data_program/', help="the embedding dimension")
    # parser.add_argument('--max_seq_length', type=int, default=50, help="the embedding dimension")
    # parser.add_argument('--layer_num', type=int, default=3, help="the embedding dimension")
    parser.add_argument('--voting', default=False, action="store_true", help="the embedding dimension")
    parser.add_argument('--id', default="0", type=str, help="the embedding dimension")
    parser.add_argument('--analyze', default=False, action="store_true", help="the embedding dimension")
    #parser.add_argument('--num_epoch', type=int, default=10, help="the number of epochs for training")
    parser.add_argument('--threshold', type=float, default=0.5, help="the threshold for the prediction")
    parser.add_argument("--output_dir", default="checkpoints/", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    # parser.add_argument("--learning_rate", default=5e-4, type=float, help="The initial learning rate for Adam.")
    args = parser.parse_args()
    return args


# torch.autograd.set_detect_anomaly(True)

args = parse_opt()


if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

with open('../preprocessed_data_program/vocab.json') as f:
    vocab = json.load(f)


start_time = time.time()
if args.do_train:
    train_examples = get_batch(option='train', data_dir=args.data_dir, vocab=vocab,
                               max_seq_length=max_seq_length, cutoff=cutoff)
    train_data = TensorDataset(*train_examples)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

if args.do_val:
    val_examples = get_batch(option='val', data_dir=args.data_dir, vocab=vocab, max_seq_length=max_seq_length)
    val_data = TensorDataset(*val_examples)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

if args.do_test:
    val_examples = get_batch(option='test', data_dir=args.data_dir, vocab=vocab,  max_seq_length=max_seq_length)
    val_data = TensorDataset(*val_examples)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

if args.do_simple_test:
    val_examples = get_batch(option='simple_test', data_dir=args.data_dir,
                             vocab=vocab,  max_seq_length=max_seq_length)
    val_data = TensorDataset(*val_examples)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

if args.do_complex_test:
    val_examples = get_batch(option='complex_test', data_dir=args.data_dir,
                             vocab=vocab,  max_seq_length=max_seq_length)
    val_data = TensorDataset(*val_examples)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

if args.do_small_test:
    val_examples = get_batch(option='small_test', data_dir=args.data_dir,
                             vocab=vocab,  max_seq_length=max_seq_length)
    val_data = TensorDataset(*val_examples)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

print("Loading used {} secs".format(time.time() - start_time))

# encoder_stat = Encoder(vocab_size=len(vocab), d_word_vec=emb_dim, n_layers=layer_num, d_model=emb_dim, n_head=n_heads)
encoder_stat = Encoder(vocab_size_in=len(vocab), vocab_size_out=2, d_model=emb_dim, n_heads=n_heads, num_layers=layer_num, dropout=dropout, device=device)
# encoder_prog = Decoder(vocab_size=len(vocab), d_word_vec=emb_dim, n_layers=layer_num, d_model=emb_dim, n_head=n_heads)

encoder_stat.to(device)
# encoder_prog.to(device)
# classifier.to(device)


def evaluate(val_dataloader, encoder_stat, cutoff_val):

    device = encoder_stat.device
    mapping = {}
    TP, TN, FN, FP = 0, 0, 0, 0
    
    with torch.no_grad():  # Disable gradient computation for faster evaluation
        for val_step, batch in enumerate(val_dataloader):
            if val_step > cutoff_val and cutoff_val > 0:
              break
            batch = tuple(t.to(device) for t in batch)
            input_ids, prog_ids, labels, index, true_lab, pred_lab = batch

            logits = encoder_stat(torch.cat((input_ids, prog_ids), dim=-1))[:, -1, :]
            sigx = logits[:, 1] - logits[:, 0]
            similarity = torch.sigmoid(sigx)
            
            # similarity = similarity.cpu().data.numpy()
            sim = (similarity > args.threshold).float()
            # labels = labels.cpu().data.numpy()
            # print("Ones fraction:", labels.sum()/len(labels))
            # print(labels.shape)
            # sys.exit()
            # index = index.cpu().data.numpy()
            # true_lab = true_lab.cpu().data.numpy()
            # pred_lab = pred_lab.cpu().data.numpy()

            TP += ((sim == 1) & (labels == 1)).sum()
            TN += ((sim == 0) & (labels == 0)).sum()
            FN += ((sim == 0) & (labels == 1)).sum()
            FP += ((sim == 1) & (labels == 0)).sum()

            # Simple mapping for per-example tracking
            if not args.voting:
                for i, s, p, t in zip(index, similarity, pred_lab, true_lab):
                    i = i.item()  # Convert index to Python int for dictionary key
                    if i not in mapping:
                        mapping[i] = [s.item(), p.item(), t.item()]
                    else:
                        if s.item() > mapping[i][0]:
                            mapping[i] = [s.item(), p.item(), t.item()]
            else:
                factor = 2
                for i, s, p, t in zip(index, similarity, pred_lab, true_lab):
                    i = i.item()  # Convert index to Python int for dictionary key
                    if i not in mapping:
                        if p == 1:
                            mapping[i] = [factor * s.item(), s.item(), t.item()]
                        else:
                            mapping[i] = [-s.item(), s.item(), t.item()]
                    else:
                        if p == 1:
                            mapping[i][0] += factor * s.item()
                        else:
                            mapping[i][0] -= s.item()

    precision = TP / (TP + FP + 0.001)
    recall = TP / (TP + FN + 0.001)
    print("TP: {}, FP: {}, FN: {}, TN: {}. precision = {}: recall = {}".format(TP, FP, FN, TN, precision, recall))

    # Calculate accuracy from mapping
    if not args.voting:
        success, fail = 0, 0
        for i, line in mapping.items():
            if line[1] == line[2]:
                success += 1
            else:
                fail += 1
        print("success = {}, fail = {}, accuracy = {}".format(success, fail, success / (success + fail + 0.001)))
        accuracy = success / (success + fail + 0.001)
    else:
        success, fail = 0, 0
        for i, ent in mapping.items():
            if (ent[0] > 0 and ent[2] == 1) or (ent[0] < 0 and ent[2] == 0):
                success += 1
            else:
                fail += 1
        print("success = {}, fail = {}, accuracy = {}".format(success, fail, success / (success + fail + 0.001)))
        accuracy = success / (success + fail + 0.001)
    
    return precision, recall, accuracy




def evaluate_quantize(val_dataloader, encoder_stat, cutoff_val, m_max, e_max):

    device = encoder_stat.device
    mapping = {}
    TP, TN, FN, FP = 0, 0, 0, 0
    
    with torch.no_grad():  # Disable gradient computation for faster evaluation
        for val_step, batch in enumerate(val_dataloader):
            if val_step > cutoff_val and cutoff_val > 0:
              break
            batch = tuple(t.to(device) for t in batch)
            input_ids, prog_ids, labels, index, true_lab, pred_lab = batch

            logits = encoder_stat(torch.cat((input_ids, prog_ids), dim=-1), quantization=True, m_max=m_max, e_max=e_max)[:, -1, :]
            sigx = logits[:, 1] - logits[:, 0]
            similarity = torch.sigmoid(sigx)
            
            # similarity = similarity.cpu().data.numpy()
            sim = (similarity > args.threshold).float()
            # labels = labels.cpu().data.numpy()
            # print("Ones fraction:", labels.sum()/len(labels))
            # print(labels.shape)
            # sys.exit()
            # index = index.cpu().data.numpy()
            # true_lab = true_lab.cpu().data.numpy()
            # pred_lab = pred_lab.cpu().data.numpy()

            TP += ((sim == 1) & (labels == 1)).sum()
            TN += ((sim == 0) & (labels == 0)).sum()
            FN += ((sim == 0) & (labels == 1)).sum()
            FP += ((sim == 1) & (labels == 0)).sum()

            # Simple mapping for per-example tracking
            if not args.voting:
                for i, s, p, t in zip(index, similarity, pred_lab, true_lab):
                    i = i.item()  # Convert index to Python int for dictionary key
                    if i not in mapping:
                        mapping[i] = [s.item(), p.item(), t.item()]
                    else:
                        if s.item() > mapping[i][0]:
                            mapping[i] = [s.item(), p.item(), t.item()]
            else:
                factor = 2
                for i, s, p, t in zip(index, similarity, pred_lab, true_lab):
                    i = i.item()  # Convert index to Python int for dictionary key
                    if i not in mapping:
                        if p == 1:
                            mapping[i] = [factor * s.item(), s.item(), t.item()]
                        else:
                            mapping[i] = [-s.item(), s.item(), t.item()]
                    else:
                        if p == 1:
                            mapping[i][0] += factor * s.item()
                        else:
                            mapping[i][0] -= s.item()

    precision = TP / (TP + FP + 0.001)
    recall = TP / (TP + FN + 0.001)
    print("TP: {}, FP: {}, FN: {}, TN: {}. precision = {}: recall = {}".format(TP, FP, FN, TN, precision, recall))

    # Calculate accuracy from mapping
    if not args.voting:
        success, fail = 0, 0
        for i, line in mapping.items():
            if line[1] == line[2]:
                success += 1
            else:
                fail += 1
        print("success = {}, fail = {}, accuracy = {}".format(success, fail, success / (success + fail + 0.001)))
        accuracy = success / (success + fail + 0.001)
    else:
        success, fail = 0, 0
        for i, ent in mapping.items():
            if (ent[0] > 0 and ent[2] == 1) or (ent[0] < 0 and ent[2] == 0):
                success += 1
            else:
                fail += 1
        print("success = {}, fail = {}, accuracy = {}".format(success, fail, success / (success + fail + 0.001)))
        accuracy = success / (success + fail + 0.001)
    
    return precision, recall, accuracy









if args.resume:
    encoder_stat.load_state_dict(torch.load(args.output_dir + "encoder_stat_{}.pt".format(args.id)))
    # encoder_prog.load_state_dict(torch.load(args.output_dir + "encoder_prog_{}.pt".format(args.id)))
    #classifier.load_state_dict(torch.load(args.output_dir + "classifier.pt"))
    print("Reloading saved model {}".format(args.output_dir))

if args.do_train:
    loss_func = torch.nn.BCEWithLogitsLoss(reduction="mean")
    loss_func.to(device)
    encoder_stat.train()
    # encoder_prog.train()
    # classifier.train()
    print("Start Training with {} batches".format(len(train_dataloader)))

    # params = chain(encoder_stat.parameters(), encoder_prog.parameters())
    params = encoder_stat.parameters()
    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, params),
                                 lr=learning_rate, betas=(0.9, 0.98), eps=0.9e-09)
    best_accuracy = 0
    for epoch in range(num_epoch):
        print(f"Epoch: {epoch}" + "="*20)
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, prog_ids, labels, index, true_lab, pred_lab = batch

            encoder_stat.zero_grad()
            # encoder_prog.zero_grad()
            optimizer.zero_grad()

            logits = encoder_stat(torch.cat((input_ids, prog_ids), dim=-1))[:, -1, :]
            # enc_prog, logits = encoder_prog(prog_ids, input_ids, enc_stat)
            """
			mag_stat = torch.norm(enc_stat, p=2, dim=1)
			mag_prog = torch.norm(enc_prog, p=2, dim=1)
			similarity = (enc_stat * enc_prog).sum(-1) / (mag_stat * mag_prog)

			loss = -torch.mean(torch.log(similarity))
			"""
            # loss = loss_func(logits, labels)
            loss = F.cross_entropy(logits, labels.long())

            similarity = torch.sigmoid(logits)
            pred = (similarity > args.threshold).float()

            loss.backward()
            optimizer.step()

            if (step + 1) % 5 == 0:
                print("Loss function = {}".format(loss.item()))

            if (step + 1) % 5 == 0:
                # print(step)
                encoder_stat.eval()
                # encoder_prog.eval()
                # classifier.eval()

                precision, recall, accuracy = evaluate(val_dataloader, encoder_stat, cutoff_val)

                if accuracy > best_accuracy:
                    torch.save(encoder_stat.state_dict(), args.output_dir + "encoder_stat_{}.pt".format(args.id))
                    # torch.save(encoder_prog.state_dict(), args.output_dir + "encoder_prog_{}.pt".format(args.id))
                    #torch.save(classifier.state_dict(), args.output_dir + "classifier.pt")
                    best_accuracy = accuracy

                encoder_stat.train()
                # encoder_prog.train()
                # classifier.train()


  

if args.do_val or args.do_test or args.do_simple_test or args.do_complex_test or args.do_small_test:
    encoder_stat.eval()
    # encoder_prog.eval()
    # classifier.eval()
    precision, recall, accuracy = evaluate(val_dataloader, encoder_stat, cutoff_val)
    print("Accuracy:", accuracy)

    t, e = 24, 8
    t, e = 5, 3


    precision, recall, accuracy = evaluate_quantize(val_dataloader, encoder_stat, cutoff_val, t, e)
    
    print("Quantized Accuracy:", accuracy)
    # # Move to CPU and convert to float16
    # encoder_stat = encoder_stat.to('cpu', dtype=torch.float32)  # First ensure float32 on CPU
    # encoder_stat.update_dtype_device(dtype=torch.float32, device=torch.device('cpu'))
    
    # # Now convert to float16
    # encoder_stat = encoder_stat.half()  # Use half() to convert all parameters to float16
    # encoder_stat.update_dtype_device(dtype=torch.float16, device=torch.device('cpu'))
    
    # precision, recall, accuracy = evaluate(val_dataloader, encoder_stat, cutoff_val)
    # print("Accuracy with float16:", accuracy)


