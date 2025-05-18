CUDA_VISIBLE_DEVICES=2 python finetune.py \
    --model_name_or_path facebook/detr-resnet-50 \
    --output_dir ckpts/detr-finetuned-combined-1-500-epochs \
    --dataset_name cppe-5 \
    --dataset_path ./workspace/iva/data/combined \
    --do_train true \
    --do_eval true \
    --num_train_epochs 500 \
    --image_square_size 600 \
    --fp16 true \
    --learning_rate 1e-4 \
    --lr_backbone 1e-5 \
    --weight_decay 1e-4 \
    --lr_scheduler_type linear \
    --dataloader_num_workers 8 \
    --dataloader_prefetch_factor 4 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --remove_unused_columns false \
    --eval_do_concat_batches false \
    --ignore_mismatched_sizes true \
    --metric_for_best_model eval_map \
    --greater_is_better true \
    --load_best_model_at_end true \
    --logging_strategy epoch \
    --eval_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 2 \
    --push_to_hub false \
    --push_to_hub_model_id cdetr-finetuned-4-10k-steps \
    --hub_strategy end \
    --seed 2025


# --overwrite_output_dir

# facebook/detr-resnet-50