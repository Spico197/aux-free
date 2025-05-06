export CUDA_VISIBLE_DEVICES="6"
GPUS=1
PORT=$((1024 + RANDOM % 64511))

torchrun --standalone --nproc_per_node=$GPUS --master_port $PORT \
    -m src.train \
    --output_dir ./output/bal_0.001-auxfree_0.001 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --max_train_steps 500 \
    --with_tracking \
    --mixed_precision fp16 \
    --router_aux_loss_coef 0.001 \
    --balance_type "noaux_tc" \
    --update_rate 0.001
