import torch
import math
import tiktoken
from model import GPT
from model import GPTConfig
from DataLoader import DataLoaderLite
import time

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

print(f"using device : {device}")



torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288
B = 16
T = 1024
assert total_batch_size % (B * T) == 0
grad_accum_steps = total_batch_size // (B * T)
print(f"total desired batch size: {total_batch_size }")
print(f" => Calculated gradient accumulation step: {grad_accum_steps}") 


train_loader = DataLoaderLite(B=4, T=1024)
torch.set_float32_matmul_precision('high')


#get logits
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
#logits, loss = model(x, y)

max_lr = 6e-4
min_lr = max_lr*0.1
warmup_steps =  10
max_steps = 50

def get_lr(it):
    if it < warmup_steps : 
        return max_lr * (it +1 ) / warmup_steps
    if it > max_steps :
        return min_lr
    # belongs to [0,1]
    decay_ratio = (it - warmup_steps) / ( max_steps - warmup_steps) 
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) #starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


 

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    for param_group in optimizer.param_groups: 
        param_group['lr'] = get_lr(step)
    optimizer.step()
    t1 = time.time()
    dt = (t1 - t0)*1000
    tokens_processed = train_loader.B * train_loader.T
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"step {step}, loss : {loss_accum.item()}, lr : {get_lr(step):.4e} dt : {dt:.2f}ms")


print(logits.shape)
print(loss)
import sys; sys.exit(0)

#prefix tokens
model.eval()
num_return_sequences = 5
max_length = 30

# generate! right now x is (B, T) where B = 5, T = 8

# set the seed to 42
torch.manual_seed(42)

while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x)  # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :]  # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs, 1)  # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)