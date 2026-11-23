# Neural Networks from Scratch

Hands-on implementations of neural networks in Python and PyTorch. I wrote the
code myself, step by step, and checked each piece against PyTorch where possible.

## What I built
- **micrograd**: a small autograd engine (backpropagation from scratch)
- **MLP**: character-level language model with a manual BatchNorm
- **Deep network**: `Linear`, `Tanh`, and `BatchNorm1d` implemented as classes by hand
- **Manual backpropagation**: gradients written by hand through a network with
  BatchNorm, compared against PyTorch (in progress)
- **WaveNet-style model**: hierarchical character model (in progress)

## What I learned
- Why initialization matters: saturated tanh units stop learning.
- How BatchNorm rescales activations, and why a bug in its dimensions can
  silently give wrong statistics.
- Backprop reduces to one rule: local derivative x upstream gradient. If a
  variable is broadcast, sum over the broadcast dimension. If it feeds two
  places, add the gradients.
- Checking tensor shapes catches most bugs early.

## Status
| Part | Status |
|------|--------|
| Notebook: micrograd, MLP, BatchNorm, deep network | Done (long training run in progress) |
| Manual backpropagation (BatchNorm network) | In progress |
| WaveNet-style model | In progress |

## Background
I'm a data analyst (Google Data Analytics and Advanced Data Analytics
certificates). I'm learning deep learning so I can read and understand
PyTorch code.

## Credits
Exercises and notebook templates are from Andrej Karpathy's
"Neural Networks: Zero to Hero" course. All solutions and code were written by me.
