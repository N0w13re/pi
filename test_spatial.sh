export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
source examples/libero/.venv/bin/activate

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python examples/libero/main.py \
    --args.task_suite_name libero_spatial \
    --args.video_out_path ./results/libero_spatial \
    --args.host 127.0.0.1 \
    --args.port 8001