import tensorflow as tf
from transformers import GPT2LMHeadModel, GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("xlnet-base-cased")
model = GPT2LMHeadModel.from_pretrained("xlnet-base-cased", pad_token_id=tokenizer.eos_token_id)
