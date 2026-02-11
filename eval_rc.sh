# apt-get update
# apt-get install -y libvulkan1 libgl1-mesa-glx libegl1-mesa libx11-6 ffmpeg

# conda create -c conda-forge -n robocasa python=3.10 -y
# conda activate robocasa

# cd robosuite/
# pip install -e .
# cd ..

# cd robocasa/
# pip install -e .
# cd ..

# pip install PyOpenGL-accelerate
# pip install tyro
# pip install imageio[ffmpeg] 
# uv pip install -e packages/openpi-client

# cd robocasa/
# python robocasa/scripts/setup_macros.py 
# cd ..



# ## RUN
# cd robocasa
# python eval_client.py

# bash gr00t/eval/sim/robocasa/setup_RoboCasa.sh
# uv pip install -e packages/openpi-client


# cd /pi/Isaac-GR00T/gr00t/eval/sim/robocasa
# SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# PROJECT_REPO="$SCRIPT_DIR/../../../.."
# ROBOCASA_REPO="$PROJECT_REPO/external_dependencies/robocasa"
# UV_ENV="$SCRIPT_DIR/robocasa_uv"
# source "$UV_ENV/.venv/bin/activate"
# cd /pi

# python /pi/Isaac-GR00T/external_dependencies/robocasa/robocasa/scripts/setup_macros.py

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

cd Isaac-GR00T
# gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python gr00t/eval/sim/robocasa/robocasa_uv/.venv/lib/python3.10/site-packages/robosuite/scripts/setup_macros.py

tasks=("CloseDrawer" "CoffeePressButton" "OpenDrawer" "PnPCounterToCab" "TurnSinkSpout")

PORT=$(($1 + 8000))
IDX=$1
GPU_ID=$(($IDX + 1))
TASK=${tasks[$IDX]}

export CUDA_VISIBLE_DEVICES=$GPU_ID

# gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python gr00t/eval/rollout_policy_pi.py \
#     --n_episodes 500 \
#     --policy_client_host 127.0.0.1 \
#     --policy_client_port $PORT \
#     --max_episode_steps 720 \
#     --env_name robocasa_panda_omron/${TASK}_PandaOmron_Env \
#     --n_action_steps 8 \
#     --n_envs 10

for task in "${tasks[@]}"; do
    echo "=============================="
    echo "Running task: $task"
    echo "=============================="

    gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python gr00t/eval/rollout_policy_pi.py \
        --n_episodes 500 \
        --policy_client_host 127.0.0.1 \
        --policy_client_port $PORT \
        --max_episode_steps 720 \
        --env_name robocasa_panda_omron/${task}_PandaOmron_Env \
        --n_action_steps 8 \
        --n_envs 25

    echo "Finished task: $task"
    sleep 300
done