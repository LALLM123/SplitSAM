You can refer to the following link for the preliminary operation steps : https://github.com/FDU-INC/SplitFM

For our contribution here, you can run gpt2_ft_sfl_LETS_v1.py. 

For example, run:

```python
python -m torch.distributed.launch --nproc_per_node=1 --use_env src/gpt2_ft_sfl_LETS_v1.py \
--train_data0 ./data/webnlg_challenge_2017/train0.jsonl \
--train_data1 ./data/webnlg_challenge_2017/train1.jsonl \
--train_data2 ./data/webnlg_challenge_2017/train2.jsonl \
--valid_data ./data/webnlg_challenge_2017/valid.jsonl \
--train_batch_size 4 \
--grad_acc 1 \
--valid_batch_size 4 \
--seq_len 512 \
--model_card gpt2.sm \
--init_checkpoint ./pretrained_checkpoints/gpt2-pytorch_model.bin \
--platform local \
--clip 0.0 \
--lr 0.0002 \
--sam_rho 0.03 \
--weight_decay 0.01 \
--correct_bias \
--adam_beta2 0.999 \
--scheduler linear \
--warmup_step 500 \
--max_epoch 5 \
--save_interval 400000 \
--lora_dim 2 \
--lora_alpha 32 \
--lora_dropout 0.1 \
--label_smooth 0.1 \
--work_dir ./trained_models/GPT2_S/webnlg_challenge_2017 \
--random_seed 40
```
