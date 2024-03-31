import numpy as np
import torch
import torch.nn as nn

BLOCK_SIZE = 8  # context window size
BATCH_SIZE = 4
N_EMBD = 128   # embedding size, should = N_HEADS * HEAD_SIZE
NUM_HEAD = 4   # number of heads in multi-head attention
HEAD_SIZE = 32 # size of each head
LR = 3e-3
TRAINING_ITERS = 32000
EVAL_ITERS = 200
DROPOUT = 0.1
NUM_BLOCKS = 2 


# Load data
f = open("input.txt", "r")
raw_input_str = f.read()

# Preprocessing
chars = sorted(list(set(raw_input_str)))
vocab_size = len(chars)
ctoi={c: i for i, c in enumerate(chars)}
itoc={i: c for i, c in enumerate(chars)}
def encode(raw_str, ctoi):
  return [ctoi[c] for c in raw_str]

def decode(tokens, itoc):
  return ''.join([itoc[i] for i in tokens])

# split training and eval data
tt = torch.tensor(encode(raw_input_str, ctoi), dtype=torch.long)
training_split = tt[: int(len(tt) * 0.9)]
eval_split = tt[int(len(tt) * 0.9):]


def get_batch(
    training_data,
    eval_data,
    block_size=BLOCK_SIZE,
    batch_size=BATCH_SIZE,
    split='training',
):
  data = training_data if split == 'training' else eval_data
  block_start_idx = torch.randint(0, len(data)-block_size-1, (batch_size,))
  X = torch.stack([data[idx:idx+block_size] for idx in block_start_idx])
  Y = torch.stack([data[idx+1:idx+block_size+1] for idx in block_start_idx])
  return X, Y

def evaluate_loss(model):
  eval_batches=EVAL_ITERS
  losses = {}
  model.eval()
  for split in ['training', 'eval']:
    loss = []
    for i in range(eval_batches):
      xb, yb = get_batch(training_split, eval_split, split=split)
      _, l = model(xb, yb)
      loss.append(l.item())
    losses[split]=sum(loss)/len(loss)
  model.train()
  return losses



class Head(nn.Module):
  def __init__(self, n_embd, head_size):
    super().__init__()
    self.query = nn.Linear(n_embd, head_size)
    self.key = nn.Linear(n_embd, head_size)
    self.value = nn.Linear(n_embd, head_size)
    self.register_buffer('tril', torch.tril(torch.ones((BLOCK_SIZE, BLOCK_SIZE))))
    self.dropout = nn.Dropout(DROPOUT)

  def forward(self, x):
    B, T, C = x.shape 
    q = self.query(x) 
    k = self.key(x)
    v = self.value(x)
    wei = q @ k.transpose(-1, -2) * k.shape[-1]**(-0.5) 
    mask = self.tril[:T, :T]
    wei = wei.masked_fill(mask==0, float('-inf'))
    wei = nn.functional.softmax(wei, dim=-1)
    wei = self.dropout(wei)
    return wei @ v
  
class MultiHead(nn.Module):
  def __init__(self, n_embd, head_size, num_heads):
    super().__init__()
    self.heads = [Head(n_embd=n_embd, head_size=head_size) for i in range(num_heads)]
    self.proj = nn.Linear(num_heads * head_size, n_embd)
    self.dropout = nn.Dropout(DROPOUT)

  def forward(self, x):
    x = torch.cat([h(x) for h in self.heads], dim=-1)
    x = self.dropout(self.proj(x))
    return x

class FeedForward(nn.Module):
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embd, 4*n_embd),
        nn.ReLU(),
        nn.Linear(4*n_embd, n_embd),
        nn.Dropout(DROPOUT),
    )
  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  def __init__(self,n_embd, head_size, num_heads):
    super().__init__()
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)
    self.att = MultiHead(n_embd, head_size, num_heads)
    self.fwd = FeedForward(n_embd)
  
  def forward(self, x):
    x = x + self.att(self.ln1(x))
    x = x + self.ln2(self.fwd(x))
    return x



class GPTModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.embedding_table=nn.Embedding(vocab_size, N_EMBD)
    self.pos_embedding_table = nn.Embedding(vocab_size, N_EMBD)
    self.blocks = [Block(N_EMBD, HEAD_SIZE, NUM_HEAD) for i in range(NUM_BLOCKS)]
    self.ln_f = nn.LayerNorm(N_EMBD)
    self.lm = nn.Linear(N_EMBD, vocab_size)

  def forward(self, input_idx, target=None):
    x = self.embedding_table(input_idx) + self.pos_embedding_table(input_idx)
    logits = nn.Sequential(*self.blocks, self.ln_f, self.lm)(x)
    loss = None
    B, T, C = logits.shape 
    if target is not None:
      loss = nn.functional.cross_entropy(logits.view(B*T, C), target.view(B*T))
    return logits, loss
      


  def generate(self, input_idx, max_num_samples):
    idx = input_idx # input_idx # (B, T)
    for i in range(max_num_samples):
      idx_block = idx[:,-BLOCK_SIZE:]
      logits,_ = self.forward(idx_block) # (B, T, C)
      next_token_logits = logits[:,-1,:] #(B, C)
      next_token_prob = nn.functional.softmax(next_token_logits, dim=-1) # (B, C)
      next_token = torch.multinomial(next_token_prob, 1, replacement=True) #(B, 1)
      idx = torch.cat((idx, next_token), dim=-1)
    return idx
  


model = GPTModel(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

for i in range(TRAINING_ITERS):
  xb, yb = get_batch(training_split, eval_split)
  logtis, loss = model.forward(xb, yb)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  if i%500==0:
    print(evaluate_loss(model))