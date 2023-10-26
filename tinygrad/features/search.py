from typing import Dict, List, cast, DefaultDict, Optional
from tinygrad.lazy import vars_from_ast
from tinygrad.ops import Device, Compiled, MemBuffer
from tinygrad.helpers import prod, getenv, ImageDType, flatten, DEBUG
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.runtime.lib import RawBuffer
from collections import defaultdict

from tinygrad.codegen.optimizer import Opt, OptOps
actions = flatten([[Opt(op=OptOps.UPCAST, axis=axis, amt=amt) for amt in [0,2,3,4,7]] for axis in range(6)])
actions += flatten([[Opt(op=OptOps.UNROLL, axis=axis, amt=amt) for amt in [0,4]] for axis in range(4)])
actions += flatten([[Opt(op=OptOps.LOCAL, axis=axis, amt=amt) for amt in [2,3,4,8,16]] for axis in range(5)])
actions += [
  Opt(op=OptOps.LOCAL, axis=0, amt=32),
  Opt(op=OptOps.GROUP, axis=1, amt=4), Opt(op=OptOps.GROUP, axis=1, amt=8), Opt(op=OptOps.GROUP, axis=2, amt=8),
  Opt(op=OptOps.GROUPTOP, axis=0, amt=16), Opt(op=OptOps.GROUPTOP, axis=0, amt=256),
  Opt(op=OptOps.GROUPTOP, axis=1, amt=16), Opt(op=OptOps.GROUPTOP, axis=1, amt=256),
  Opt(op=OptOps.GROUPTOP, axis=2, amt=16), Opt(op=OptOps.GROUPTOP, axis=2, amt=256)
]

# returns time in seconds
import shelve
logtm = shelve.open(getenv("LOGTM", "")) if getenv("LOGTM", "") else None
step_cache = shelve.open(getenv('STEP_CACHE', './step_cache'))


import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
  def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, kernel_sizes, output_dim, dropout, pretrained_embeddings=None):
    super(TextCNN, self).__init__()
    
    if pretrained_embeddings is None:
      self.embedding = nn.Embedding(vocab_size, embedding_dim)
    else:
      self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
    
    self.convs = nn.ModuleList([
      nn.Sequential(
        nn.Conv1d(in_channels=embedding_dim, out_channels=fs, kernel_size=k),
        nn.BatchNorm1d(fs),
        nn.ReLU()
      )
      for fs in filter_sizes for k in kernel_sizes
    ])
      
    self.fcs1 = nn.ModuleList([nn.Linear(fs, output_dim*2) for fs in filter_sizes for k in kernel_sizes])
    self.fc2 = nn.Linear(output_dim*2, output_dim)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, text, text_lens):
    embedded = self.embedding(text)
    embedded = embedded.permute(0, 2, 1)
    
    conved = [conv(embedded) for conv in self.convs]
    
    # Using both max and average pooling
    max_pooled = [self.fcs1[i](F.max_pool1d(conv, conv.shape[2]).squeeze(2)) for i, conv in enumerate(conved)]
    avg_pooled = [self.fcs1[i](F.avg_pool1d(conv, conv.shape[2]).squeeze(2)) for i, conv in enumerate(conved)]
    fc1_out = F.relu(sum(max_pooled + avg_pooled))
    fc1_out = self.dropout(fc1_out)
    return self.fc2(fc1_out)

import torch
import pickle
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file('ast_tokenizer.tok')
m = TextCNN(366, 32, 4, (30,), (3,), 60, 0.3)
m.load_state_dict(torch.load('policynet.bin'))
m = m.eval()
    
with open('mapping.pkl', 'rb') as f:
  vocab = pickle.load(f)
  label_to_action = pickle.load(f)
action_to_label = {v:k for k,v in label_to_action.items()}


import string
import code_tokenize as ctok

def pre_tokenize(code):
  tokens = []
  raw = ctok.tokenize(str(code[0]), lang="python") + ctok.tokenize(str(code[1]), lang="python")
  for tok in raw:
    if str(tok)[-1].isdigit():
      if '.' not in str(tok):
        tokens.append(str(tok))
    else:
      if str(tok) not in string.punctuation:
        tokens.append(str(tok))
  return tokens

def tokenize(k):
  return tokenizer.encode(' '.join(list(map(str,pre_tokenize(k)))))


def time_linearizer(lin:Linearizer, rawbufs:List[RawBuffer], allow_test_size=True, max_global_size=65536, cnt=3, should_copy=True, disable_cache=False) -> float:
  key = str((lin.ast, lin.applied_opts))
  if should_copy and not disable_cache and logtm is not None and key in logtm: return min(logtm[key])  # pylint: disable=E1135 # NOTE: we check should_copy since this may have side effects
  if should_copy: lin = lin.copy() # TODO: remove the need for this
  var_vals = {k:k.min for k in vars_from_ast(lin.ast)}
  try:
    lin.linearize()
    device_compiler = cast(Compiled, Device[Device.DEFAULT])
    prg = device_compiler.to_program(device_compiler.to_code(lin))
    real_global_size = prg.global_size
    if allow_test_size and prg.global_size:
      test_global_size = prg.global_size[:]
      while prod(test_global_size) > max_global_size:
        for j in range(2,-1,-1):
          if test_global_size[j] > 16:
            test_global_size[j] //= 2
            break
      factor = prod(prg.global_size) / prod(test_global_size)
      prg.global_size = test_global_size
      #print(real_global_size, test_global_size, factor)
    else:
      factor = 1
    # TODO: this is super broken for var_vals
    # TODO: this is copied from prg.__call__
    global_size, local_size = prg.launch_dims(var_vals)
    if global_size is not None and local_size is None:
      local_size = prg.optimize_local_size(global_size, rawbufs)
      global_size = [g//l if g%l == 0 else g/l for g,l in zip(global_size, local_size)]
    tms = [prg.clprg(global_size, local_size, *rawbufs, *var_vals.values(), wait=True)*factor for _ in range(cnt)]
    prg.global_size = real_global_size
  except Exception:
    #import traceback; traceback.print_exc()
    #print("FAILED")
    #print(lin.ast)
    #print(lin.applied_opts)
    tms = [float('inf')]
  if logtm is not None: logtm[key] = tms
  return min(tms)

# get (scrap) buffers for timing the linearizer
def bufs_from_lin(lin:Linearizer) -> List[RawBuffer]:
  bufsts:DefaultDict[int, List[MemBuffer]] = defaultdict(list)
  for x in lin.membufs: bufsts[x.idx].append(x)
  rawbufs:List[Optional[RawBuffer]] = [None]*len(bufsts)
  for k,lx in bufsts.items():
    rawbufs[k] = cast(Compiled, Device[Device.DEFAULT]).buffer(prod(lx[0].dtype.shape) if isinstance(lx[0].dtype, ImageDType) else max(y.st.size() for y in lx), lx[0].dtype)
  assert all(r is not None for r in rawbufs)
  return cast(List[RawBuffer], rawbufs)

# get dictionary of all possible actions
def get_linearizer_actions(lin:Linearizer, include_0=True) -> Dict[int, Linearizer]:
  acted_lins = {0:lin.copy()} if include_0 else {}
  for i,a in enumerate(actions):
    if a.axis >= lin.shape_len: continue
    if lin.full_shape[a.axis] == a.amt and Opt(a.op, a.axis, 0) in actions: continue
    lin2 = lin.copy()
    try:
      lin2.apply_opt(a)
      up, lcl = 1, 1
      for s,c in zip(lin2.full_shape, lin2.colors()):
        if c in {"magenta", "yellow"}: up *= s
        if c in {"cyan", "green", "white"}: lcl *= s
      if up > 256 or lcl > 256: continue
      acted_lins[i+1] = lin2
    except Exception:
      pass
  return acted_lins

def predict_policy(lin:Linearizer, top_policies):
  acted_lins = {}
  ast, applied_opts = lin.ast, lin.applied_opts
  with torch.no_grad():
    sft = torch.nn.Softmax(1)
    tokens = torch.tensor(tokenize((ast, [action_to_label.get(str(opt), action_to_label[None]) for opt in applied_opts])).ids[-200:])
    raw = dict(zip(label_to_action.values(), sft(m(tokens.unsqueeze(0), None)).squeeze(0).cpu().numpy()))
  pred_ops = {eval(str(op)): prob for op,prob in raw.items()}
  model_acts = [act[0] for act in sorted(pred_ops.items(), key=lambda x:-x[1])]
  added = 0
  for i, a in enumerate(model_acts):
    if a is None:
      acted_lins[i+1] = lin.copy()
      continue
    if added >= top_policies:
      break
    if a.axis >= lin.shape_len: continue
    if lin.full_shape[a.axis] == a.amt and Opt(a.op, a.axis, 0) in actions: continue
    lin2 = lin.copy()
    try:
      lin2.apply_opt(a)
      up, lcl = 1, 1
      for s,c in zip(lin2.full_shape, lin2.colors()):
        if c in {"magenta", "yellow"}: up *= s
        if c in {"cyan", "green", "white"}: lcl *= s
      if up > 256 or lcl > 256: continue
      acted_lins[i+1] = lin2
      added += 1
    except Exception:
      pass
  return acted_lins


def beam_search(lin: Linearizer, rawbufs, amt):
  best_tm = float('inf')
  beam: List[Linearizer] = [lin]
  while 1:
    acted_lins = flatten([predict_policy(lin, top_policies=amt).values() for lin in beam])
    timed_lins = [(v,time_linearizer(v, rawbufs)) for v in acted_lins]
    opts = sorted(timed_lins, key=lambda x: x[1])
    if len(opts) == 0 or best_tm <= opts[0][1]: break  # we didn't get faster
    best_tm = opts[0][1]
    beam = [x[0] for x in opts[:amt]]
    # step_cache[str(beam[0].ast) + '\n\n' + str(beam[0].applied_opts[:-1])] = str(beam[0].applied_opts[-1])
    if DEBUG >= 1: print(f"{opts[0][1]*1e6:12.2f} us from {len(opts):3d} actions", beam[0].colored_shape())
  return beam[0]
