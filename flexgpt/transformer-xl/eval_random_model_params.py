# coding: utf-8
import argparse
import time
import math
import os, sys

import torch

from data_utils import get_lm_corpus
from mem_transformer import MemTransformerLM
from utils.exp_utils import get_logger
from utils.exp_utils import create_exp_dir
from datetime import date
from torch.profiler import profile, record_function, ProfilerActivity

today = date.today()
d3 = today.strftime("%m/%d/%y")
def main(batch_size, tgt_len, ext_len, mem_len, clamp_len):
    ################ CONFIG ##############################
    MODEL_DIR = "./models"
    OUTPUT_DIR = "./logs"

    ######################################################

    parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
    parser.add_argument('--data', type=str, default='../data/wikitext-103',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='wt103',
                        choices=['wt103', 'lm1b', 'enwik8', 'text8'],
                        help='dataset name')
    parser.add_argument('--split', type=str, default='all',
                        choices=['all', 'valid', 'test'],
                        help='which split to evaluate')
    # parser.add_argument('--batch_size', type=int, default=1,
    #                     help='batch size')
    batch_size = batch_size
    # parser.add_argument('--tgt_len', type=int, default=5,
    #                     help='number of tokens to predict')
    tgt_len = tgt_len
    # parser.add_argument('--ext_len', type=int, default=0,
    #                     help='length of the extended context')
    ext_len = ext_len
    # parser.add_argument('--mem_len', type=int, default=0,
    #                     help='length of the retained previous heads')
    mem_len = mem_len
    # parser.add_argument('--clamp_len', type=int, default=-1,
    #                     help='max positional embedding index')
    clamp_len = clamp_len
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    # parser.add_argument('--work_dir', type=str, required=True,
    #                     help='path to the work_dir')
    parser.add_argument('--no_log', action='store_true',
                        help='do not log the eval result')
    parser.add_argument('--same_length', action='store_true',
                        help='set same length attention with masking')
    args = parser.parse_args()
    assert ext_len >= 0, 'extended context length must be non-negative'

    device = torch.device("cuda" if args.cuda else "cpu")

    # Get logger
    config_to_print = 'bsz={}-tgt_len={}-ext_len={}-mem_len={}-clamp_len={}'.format(
        batch_size, tgt_len, ext_len, mem_len, clamp_len)
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, time.strftime('%Y%m%d-%H%M%S'))
    logging = create_exp_dir(OUTPUT_DIR, scripts_to_save=['train.py', 'mem_transformer.py'])
    logging = get_logger(os.path.join(OUTPUT_DIR, config_to_print + 'log.txt'),
                        log_=not args.no_log)
                        

    # Load dataset
    corpus = get_lm_corpus(args.data, args.dataset)
    ntokens = len(corpus.vocab)

    va_iter = corpus.get_iterator('valid', batch_size, tgt_len,
        device=device, ext_len=ext_len)
    te_iter = corpus.get_iterator('test', batch_size, tgt_len,
        device=device, ext_len=ext_len)


    # Load the best saved model.
    with open(os.path.join(MODEL_DIR, 'model.pt'), 'rb') as f:
        model = torch.load(f)
    model.backward_compatible()
    model = model.to(device)

    logging('Evaluating with bsz {} tgt_len {} ext_len {} mem_len {} clamp_len {}'.format(
        batch_size, tgt_len, ext_len, mem_len, clamp_len))

    model.reset_length(tgt_len, ext_len, mem_len)
    if clamp_len > 0:
        model.clamp_len = clamp_len
    if args.same_length:
        model.same_length = True

    ###############################################################################
    # Evaluation code
    ###############################################################################
    def evaluate(eval_iter):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_len, total_loss = 0, 0.
        start_time = time.time()
        with torch.no_grad():
            mems = tuple()
            for idx, (data, target, seq_len) in enumerate(eval_iter):
                ret = model(data, target, *mems)
                loss, mems = ret[0], ret[1:]
                loss = loss.mean()
                total_loss += seq_len * loss.item()
                total_len += seq_len
                # print(idx)
                if idx % 1000 == 0:
                    print(idx)
            total_time = time.time() - start_time
        logging('Time : {:.2f}s, {:.2f}ms/segment'.format(
                total_time, 1000 * total_time / (idx+1)))
        return total_loss / total_len

    # Run on test data.
    for _ in range(3):
        if args.split == 'all':
            test_loss = evaluate(te_iter)
            valid_loss = evaluate(va_iter)
        elif args.split == 'valid':
            valid_loss = evaluate(va_iter)
            test_loss = None
        elif args.split == 'test':
            test_loss = evaluate(te_iter)
            valid_loss = None


    def format_log(loss, split):
        if args.dataset in ['enwik8', 'text8']:
            log_str = '| {0} loss {1:5.2f} | {0} bpc {2:9.5f} '.format(
                split, loss, loss / math.log(2))
        else:
            log_str = '| {0} loss {1:5.2f} | {0} ppl {2:9.3f} '.format(
                split, loss, math.exp(loss))
        return log_str

    log_str = ''
    if valid_loss is not None:
        log_str += format_log(valid_loss, 'valid')
    if test_loss is not None:
        log_str += format_log(test_loss, 'test')

    logging('=' * 100)
    logging(log_str)
    logging('=' * 100)


if __name__ == "__main__":
    batch_size = 1
    tgt_len = 1000 # number of tokens to predict
    ext_lens = [0, 32, 128, 512] # length of the extended context
    mem_lens = [0, 16, 256, 1024]
    clamp_len = -1 # max positional embedding index
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            main(batch_size, tgt_len, 0, 256, clamp_len)
            print("######################################################################")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
            # for mem_len in mem_lens:
            #     for ext_len in ext_lens:    
            #         main(batch_size, tgt_len, ext_len, mem_len, clamp_len)
            