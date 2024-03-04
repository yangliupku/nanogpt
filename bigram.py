import numpy as np
import torch
import torch.nn as nn

BLOCK_SIZE = 8
BATCH_SIZE = 4
LR = 3e-3
TRAINING_ITERS = 32000
EVAL_ITERS = 200

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

# Model
class BigramModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.embedding_table=nn.Embedding(vocab_size, vocab_size)
  
  def forward(self, input_idx, target=None):
    logits = self.embedding_table(input_idx) #(B, T, C)
    loss = None 
    if target is not None: 
      B, T, C = logits.size()
      loss = nn.functional.cross_entropy(logits.view(B*T, C),target.view(B*T))
    return logits, loss

  def generate(self, input_idx, max_num_samples):
    idx = input_idx # input_idx # (B, T)
    for i in range(max_num_samples):
      logits,_ = self.forward(idx) # (B, T, C)
      next_token_logits = logits[:,-1,:] #(B, C)
      next_token_prob = nn.functional.softmax(next_token_logits, dim=-1) # (B, C)
      next_token = torch.multinomial(next_token_prob, 1, replacement=True) #(B, 1)
      idx = torch.cat((idx, next_token), dim=-1)
    return idx
    
model = BigramModel(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

for i in range(TRAINING_ITERS):
  xb, yb = get_batch(training_split, eval_split)
  logtis, loss = model.forward(xb, yb)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  if i%500==0:
    print(evaluate_loss(model))