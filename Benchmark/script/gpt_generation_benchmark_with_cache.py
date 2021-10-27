from transformers import GPT2LMHeadModel, GPT2Tokenizer
from causal_transformer_decoder import (
    CausalTransformerDecoder,
    CausalTransformerDecoderLayer,
)
import torch
import torch.nn as nn
import time
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
hdim = 768
nhead = 12
dim_feedforward = hdim * 4
num_layers = 12
vocab_size = 50257
output_lens = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
# mem_lens = [32, 64, 128, 512, 1024]
mem_len = 32
bsz = 16
print(f"Device used: {device}")
######################################################################################################
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)




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

# Causal decoder inference
print("Inference for Causal Decoder...")
first_token = torch.zeros((1, bsz)).long().to(device=device)
decoded_tokens = first_token
t = time.time()
times_causal_decoder = []
with torch.no_grad():
    start.record()
    cache = []
    # print("original cache: " + str(cache))
    for i in range(1, output_lens[-1] + 1):
        decoded_embeddings = embedding(decoded_tokens)
        if cache == None:
            output, cache = causal_decoder(decoded_embeddings, None, cache)
        else:
            output, cache = causal_decoder(decoded_embeddings, None, cache[-1 * mem_len:])
        # cache = cache[-1 * mem_len:]
        # print(i + ": " + cache)

        logits = to_vocab(output)
        top_indices = torch.argmax(logits, dim=-1)
        top_indices_last_token = top_indices[-1:]
        decoded_tokens = torch.cat([decoded_tokens, top_indices_last_token], dim=0)
        if i in output_lens:
            times_causal_decoder.append(time.time() - t)
    end.record()

torch.cuda.synchronize()

print(mem_len, max(output_lens), times_causal_decoder)
print("Runtime: " + str(start.elapsed_time(end)))
# print("Nb decoded tokens, time GPT2, time Causal Decoder, causal decoder / GPT2")
# for (nb_tokens, time_gpt, time_causal_decoder, ratio) in zip(
#     output_lens,
#     times_gpt,
#     times_causal_decoder,
#     np.array(times_causal_decoder) / np.array(times_gpt),
# ):
#     print(mem_len, nb_tokens, time_gpt, time_causal_decoder, ratio)
