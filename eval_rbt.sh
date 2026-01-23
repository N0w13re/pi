# apt-get update
# apt-get install -y libvulkan1 libgl1-mesa-glx libegl1-mesa libx11-6 ffmpeg

# cd RoboTwin/
# python script/update_embodiment_config_path.py
# cd ..

# cd RoboTwin/envs/curobo
# pip install -e . --no-build-isolation
# cd ../../..

cd RoboTwin/policy/pi0/

# tasks=("beat_block_hammer" "handover_mic" "place_a2b_left" "open_laptop" "place_mouse_pad")
tasks=("handover_mic" "place_a2b_left")
# tasks=("open_laptop" "place_mouse_pad")

for task in "${tasks[@]}"; do
    ./eval.sh ${task} demo_randomized pi0_base_aloha_robotwin_lora rbt_aloha_lora 42 0 29999 my_RoboTwin/aloha-agilex
done
