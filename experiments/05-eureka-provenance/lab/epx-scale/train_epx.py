# EPX config: GPT-2-small dimensions, word-level, provenance channel on.

out_dir = 'out-epx'
eval_interval = 500
eval_iters = 50
log_interval = 100

always_save_checkpoint = False
wandb_log = False

dataset = 'epx_word'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 64

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1
n_sources = 3

learning_rate = 6e-4
max_iters = 3000
lr_decay_iters = 3000
min_lr = 6e-5
beta2 = 0.99
warmup_iters = 200
