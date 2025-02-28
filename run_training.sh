eval "$(conda shell.bash hook)"
conda activate env_isaacsim

# run the homogeneous environment 4 times

for seed in 2 3 4 5
do
    python source/standalone/workflows/harl/train.py  --task Isaac-Multi-Agent-Flat-Anymal-C-Direct-v0 --video_interval 10_000 --num_envs 2048 --save_interval 10 --log_interval 10 --algorithm happo --headless --num_env_steps 2_000_000_000 --dir /home/isaacp/sharedrepos/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/anymal_c_multi_agent/models --seed $seed

#run heterogeneous 5 times

for seed in 2 3 4 5
do
    python source/standalone/workflows/harl/train.py  --task Isaac-Multi-Agent-Flat-Anymal-C-Direct-v0 --video_interval 10_000 --num_envs 2048 --save_interval 10 --log_interval 10 --algorithm happo --headless --num_env_steps 2_000_000_000 --dir /home/isaacp/sharedrepos/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/anymal_c_multi_agent/models --seed $seed
