export CUDA_VISIBLE_DEVICES="6"
GPUS=1
PORT=$((1024 + RANDOM % 64511))
DEBUG_PORT=5678
torchrun --standalone --nproc_per_node=$GPUS --master_port $PORT \
    -m debugpy --listen 0.0.0.0:$DEBUG_PORT --wait-for-client \
    -m src.train \
    --output_dir ./output \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --max_train_steps 2000 \
    --with_tracking \
    --mixed_precision fp16 \
    --router_aux_loss_coef 0 \
    --balance_type "noaux_tc"
