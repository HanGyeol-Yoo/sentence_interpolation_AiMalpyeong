lr=1e-4
lora_rank=16
lora_alpha=32
lora_trainable="query_key_value,dense,dense_h_to_4h,dense_4h_to_h"
modules_to_save="embed_in,embed_out"
lora_dropout=0.05

pretrained_model=EleutherAI/polyglot-ko-12.8b
chinese_tokenizer_path=EleutherAI/polyglot-ko-12.8b
dataset_dir=../data/train
per_device_train_batch_size=64
per_device_eval_batch_size=64
gradient_accumulation_steps=8
max_seq_length=1024
output_dir=../output
validation_file=../data/val/val.json

deepspeed_config_file=ds_zero2_no_offload.json


torchrun --nnodes 1 --nproc_per_node 4 run_clm_sft_with_peft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_eval  \
    --seed $RANDOM \
    --fp16 \
    --num_train_epochs 4 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 4 \
    --evaluation_strategy steps \
    --eval_steps 100000 \
    --save_steps 100000 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --max_seq_length ${max_seq_length} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --lora_dropout ${lora_dropout} \
    --modules_to_save ${modules_to_save} \
    --torch_dtype float16 \
    --load_in_kbits 8 \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False \
    --validation_file ${validation_file}
