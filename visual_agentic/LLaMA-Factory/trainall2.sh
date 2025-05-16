

CUDA_VISIBLE_DEVICES=3,4,5,6 llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path Qwen2.5-VL-7B-Instruct \
    --dataset okvq_cot_4_all \
    --dataset_dir ./data \
    --template qwen2_vl \
    --finetuning_type lora \
    --output_dir ./saves/Qwen2.5-VL-7B-Instruct/okvq_cot_4_all \
    --overwrite_cache \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --warmup_steps 20 \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --plot_loss \
    --report_to wandb \
    --lora_rank 32 \
    --lora_alpha 64 \
    --save_strategy epoch 

wait

