eval "$(conda shell.bash hook)"
conda activate env_isaacsim

num_runs=100_000_000

# run the homogeneous  happo environment 4 times

for seed in 2 3 4
do
    python source/standalone/workflows/harl/train.py  --task Isaac-Multi-Agent-Flat-Anymal-C-Direct-v0 --video_interval 10_000 --num_envs 2048 --save_interval 10 --log_interval 10 --algorithm happo --headless --num_env_steps $num_runs --dir /home/isaacp/sharedrepos/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/anymal_c_multi_agent/models --seed $seed
done

# run the homogeneous  mappo environment 4 times
for seed in 2 3 4
do
    python source/standalone/workflows/harl/train.py  --task Isaac-Multi-Agent-Flat-Anymal-C-Direct-v0 --video_interval 10_000 --num_envs 2048 --save_interval 10 --log_interval 10 --algorithm mappo --headless --num_env_steps $num_runs --dir /home/isaacp/sharedrepos/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/anymal_c_multi_agent/models --seed $seed
done

#run heterogeneous happo 5 times

for seed in 2 3 4
do
    python source/standalone/workflows/harl/train.py  --task Isaac-Anymal-H1-Push-Direct-v0 --video_interval 10_000 --num_envs 2048 --save_interval 10 --log_interval 10 --algorithm happo --headless --num_env_steps $num_runs --dir /home/isaacp/sharedrepos/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/h1_anymal_env/2_agent_model --seed $seed
done
#run heterogeneous mappo 5 times

for seed in 2 3 4
do
    python source/standalone/workflows/harl/train.py  --task Isaac-Anymal-H1-Push-Direct-v0 --video_interval 10_000 --num_envs 2048 --save_interval 10 --log_interval 10 --algorithm mappo --headless --num_env_steps $num_runs --dir /home/isaacp/sharedrepos/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/h1_anymal_env/2_agent_model --seed $seed
done