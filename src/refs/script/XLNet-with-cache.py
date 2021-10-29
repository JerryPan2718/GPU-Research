from transformers import XLNetConfig, XLNetModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initializing a XLNet configuration
configuration = XLNetConfig(
        vocab_size=32000,
        d_model=1024,
        n_layer=24,
        n_head=16,
        d_inner=4096,
        mem_len=512,
        reuse_len=None,
        use_mems_eval=True,
        use_mems_train=False,
        )
# Initializing a model from the configuration
model = XLNetModel(configuration)
# Accessing the model configuration
# configuration = model.config



tokenizer = GPT2Tokenizer.from_pretrained("xlnet-base-cased")
model = GPT2LMHeadModel.from_pretrained("xlnet-base-cased").to(device=device)
model.eval()

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state

print(outputs)

