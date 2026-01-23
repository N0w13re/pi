export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

uv run scripts/serve_policy.py --env LIBERO --port 8001 \
    policy:checkpoint --policy.config pi0_libero_low_mem_finetune \
    --policy.dir /pi/checkpoints/pi0_libero_low_mem_finetune/pi0_lora_128/29999
