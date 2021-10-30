from transformers import GPT2LMHeadModel, GPT2Tokenizer
from causal_transformer_decoder import (
    CausalTransformerDecoder,
    CausalTransformerDecoderLayer,
)
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity
import time
import numpy as np
import os

################################# Configuration: TO CHANGE ########################################
output_file_path = "/home/ubuntu/GPU-Research/Benchmark/Experiments/20211028-causal-transformer-decoder-script/"
hdim = 768
nhead = 12
dim_feedforward = hdim * 4
num_layers = 12
vocab_size = 50257
output_len = 1000
# empty_cache_freq = 0.1
fetch_cuda_stats_freq = 0.005
# mem_lens = [16]
mem_lens = [16]
batch_sizes = [16]
reps = 1
device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = True
print(f"Device used: {device}")

################################# Configuration: TO CHANGE ########################################
writer = SummaryWriter(log_dir='logs')
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


def gpt_generation_with_cache(hdim, nhead, num_layers, vocab_size, output_len, fetch_cuda_stats_freq, mem_len, batch_size, reps):
    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Initialization:
        # GPT2
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device=device)

        # Causal Decoder
        causal_decoder = CausalTransformerDecoder(
            CausalTransformerDecoderLayer(
                d_model=hdim, nhead=nhead, dim_feedforward=dim_feedforward,
            ),
            num_layers=num_layers,
        ).to(device=device)
        causal_decoder.eval()
        to_vocab = nn.Linear(hdim, vocab_size).to(device=device)
        to_vocab.eval()
        embedding = nn.Embedding(vocab_size, hdim).to(device=device)
        embedding.eval()

        print(
            "Num parameters causal decoder:",
            sum(p.numel() for p in causal_decoder.parameters() if p.requires_grad)
            + sum(p.numel() for p in to_vocab.parameters() if p.requires_grad)
            + sum(p.numel() for p in embedding.parameters() if p.requires_grad),
        )

        print("Inference for Causal Decoder...")
        first_token = torch.zeros((1, batch_size)).long().to(device=device)
        decoded_tokens = first_token
        t = time.time()
        cache_causal_decoder = []
        with torch.no_grad():
            cache = None
            start.record()
            with torch.cuda.amp.autocast(enabled=use_amp):
                for i in range(1, output_len + 1):
                    decoded_embeddings = embedding(decoded_tokens).to(device=device)
                    if cache == None:
                        output, cache = causal_decoder(decoded_embeddings, None, cache)
                    else:
                        output, cache = causal_decoder(decoded_embeddings, None, cache)
                        # print(type(cache))
                        cache = cache[:, -1 * mem_len:].to(device=device)
                        # cache = [c[-1 * mem_len:] for c in cache]
                        # print(type(cache))
                        print(str(i) + ": " + str(len(cache)) + "\n" + str(len(cache[0])))
                        
                    logits = to_vocab(output)
                    top_indices = torch.argmax(logits, dim=-1)
                    top_indices_last_token = top_indices[-1:]
                    decoded_tokens = torch.cat([decoded_tokens, top_indices_last_token], dim=0)
                    if i % int(fetch_cuda_stats_freq * output_len) == 0:
                        cache_causal_decoder.append(torch.cuda.memory_allocated(device=device))
                        torch.cuda.empty_cache()
                    # if i % int(empty_cache_freq * output_len) == 0:
                    #     torch.cuda.empty_cache()
            end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        
    avg_time = sum(times) / reps
    print("========Configuration========")
    print("mem_len: " + str(mem_len))
    print("output_len: " + str(output_len))
    print("batch_size: " + str(batch_size))
    print("========Runtime Summary========")
    print("Runtime: " + str(avg_time/1000))
    print("Cache: " + str(cache_causal_decoder))

    # Output text file
    # output_file_path = "/home/ubuntu/GPU-Research/Benchmark/"
    file_name = output_file_path + "mem_len=" + str(mem_len) + " output_len=" + str(output_len) + " batch_size=" + str(batch_size) + ".txt"
    print(file_name)
    outF = open(file_name, "w")
    outF.write(str(avg_time/1000))
    outF.write("\n")
    outF.write(str(cache_causal_decoder))
    outF.write("\n")


################################### Main ###########################################
for mem_len in mem_lens:
    for batch_size in batch_sizes:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("model_inference"):
                gpt_generation_with_cache(hdim, nhead, num_layers, vocab_size, output_len, fetch_cuda_stats_freq, mem_len, batch_size, reps)
                print("Finished -------------------------")
        print("######################################################################")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

