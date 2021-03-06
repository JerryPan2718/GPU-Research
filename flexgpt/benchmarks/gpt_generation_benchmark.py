from transformers import GPT2LMHeadModel, GPT2Tokenizer
from causal_transformer_decoder import (
    CausalTransformerDecoder,
    CausalTransformerDecoderLayer,
)
import torch
import torch.nn as nn
import time
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity

################################# Configuration: TO CHANGE ########################################
output_file_path = "/home/ubuntu/GPU-Research/experiments/20211030-benchmark/"
hdim = 768
nhead = 12
dim_feedforward = hdim * 4
num_layers = 12
vocab_size = 50257
output_len = 1000
# empty_cache_freq = 0.1
fetch_cuda_stats_freq = 0.005
# mem_lens = [16]
output_lens = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
mem_lens = [16]
batch_sizes = [16]
reps = 1
device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = True
print(f"Device used: {device}")

################################# Configuration: TO CHANGE ########################################
writer = SummaryWriter(log_dir='logs')
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


# Initialization:

# GPT2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device=device)
model.eval()

print(
    "Num parameters GPT-2:",
    sum(p.numel() for p in model.transformer.parameters() if p.requires_grad)
    + sum(p.numel() for p in model.lm_head.parameters() if p.requires_grad),
)  # 163037184

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
)  # 190666321

# Difference in the number of parameters is due to the encoder-decoder
# attention matrices that are still stored in the causal decoder (but not used)
# here. Each of them is around 2.3M parameters, so *12 it's around 27M params

def gpt_generation_with_cache(hdim, nhead, num_layers, vocab_size, output_len, fetch_cuda_stats_freq, mem_len, batch_size, reps):
    gpt_times = []
    # GPT-2 inference
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    print("Inference for GPT-2...")
    generated = tokenizer.encode("A")
    context = torch.tensor([generated]).to(device=device)
    past = None
    times_gpt = []
    t = time.time()
    gpt_cache = []
    start.record()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=use_amp):
            for i in range(1, output_lens[-1] + 1):
                outputs = model(context, past_key_values=past)
                token = torch.argmax(outputs.logits[-1, :])
                generated += [token.tolist()]
                context = token.unsqueeze(0)
                if i in output_lens:
                    times_gpt.append(time.time() - t)
                if i % int(fetch_cuda_stats_freq * output_len) == 0:
                    torch.cuda.empty_cache()
                    gpt_cache.append(torch.cuda.memory_allocated(device=device))
                
    end.record()
    # Waits for everything to finish running
    torch.cuda.synchronize()
    gpt_times.append(start.elapsed_time(end))

    file_name = output_file_path + "GPT-2-" + "mem_len=" + str(mem_len) + "-output_len=" + str(output_len) + "-batch_size=" + str(batch_size) + ".txt"
    print(file_name)
    outF = open(file_name, "w")
    outF.write(str(sum(gpt_times) / reps / 1000))
    outF.write("\n")
    outF.write(str(gpt_cache))
    outF.write("\n")

    # Causal decoder inference
    print("Inference for Causal Decoder...")
    first_token = torch.zeros((1, batch_size)).long().to(device=device)
    decoded_tokens = first_token
    t = time.time()
    causal_decoder_cache = []
    times_causal_decoder = []
    with torch.no_grad():
        cache = None
        with torch.cuda.amp.autocast(enabled=use_amp):
            for i in range(1, output_lens[-1] + 1):
                decoded_embeddings = embedding(decoded_tokens)
                output, cache = causal_decoder(decoded_embeddings, None, cache)
                logits = to_vocab(output)
                top_indices = torch.argmax(logits, dim=-1)
                top_indices_last_token = top_indices[-1:]
                decoded_tokens = torch.cat([decoded_tokens, top_indices_last_token], dim=0)
                if i in output_lens:
                    times_causal_decoder.append(time.time() - t)
                if i % int(fetch_cuda_stats_freq * output_len) == 0:
                    torch.cuda.empty_cache()
                    causal_decoder_cache.append(torch.cuda.memory_allocated(device=device))

    print("Nb decoded tokens, time GPT2, time Causal Decoder, causal decoder / GPT2")
    for (nb_tokens, time_gpt, time_causal_decoder, ratio) in zip(
        output_lens,
        times_gpt,
        times_causal_decoder,
        np.array(times_causal_decoder) / np.array(times_gpt),
    ):
        print(nb_tokens, time_gpt, time_causal_decoder, ratio)

    # avg_time = sum(times) / reps
    # Output text file
    file_name = output_file_path + "causal_decoder-" + "mem_len=" + str(mem_len) + "-output_len=" + str(output_len) + "-batch_size=" + str(batch_size) + ".txt"
    print(file_name)
    outF = open(file_name, "w")
    outF.write(str(sum(times_causal_decoder) / reps / 1000))
    outF.write("\n")
    outF.write(str(causal_decoder_cache))
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
        
