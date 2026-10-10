# nanoGPT config for the abstention experiment (baseline dataset).
# Calibrated run overrides: --dataset=facts_char_calib --out_dir=out-facts-calib

out_dir = 'out-facts-baseline'
eval_interval = 250
eval_iters = 100
log_interval = 50

always_save_checkpoint = False
wandb_log = False

dataset = 'facts_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1

learning_rate = 1e-3
max_iters = 2000
lr_decay_iters = 2000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100
