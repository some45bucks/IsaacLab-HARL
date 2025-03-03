eval "$(conda shell.bash hook)"
conda activate isaaclab

num_runs=11_123
num_envs=7
algo=happo
vid_interval=20000

python ./source/standalone/workflows/harl/train.py  --task Isaac-Multi-Agent-Flat-Anymal-C-Direct-v0 --video_interval $vid_interval --num_envs $num_envs --save_interval 10 --log_interval 10 --algorithm $algo --headless --num_env_steps $num_runs
python ./source/standalone/workflows/harl/train.py  --task Isaac-Anymal-H1-Push-Direct-v0 --video_interval $vid_interval --num_envs $num_envs --save_interval 10 --log_interval 10 --algorithm $algo  --headless --num_env_steps $num_runs
python ./source/standalone/workflows/harl/train.py  --task Isaac-Anymal-H1-Piano-Direct-v0 --video_interval $vid_interval --num_envs $num_envs --save_interval 10 --log_interval 10 --algorithm $algo  --headless --num_env_steps $num_runs
python ./source/standalone/workflows/harl/train.py  --task Isaac-Shadow-Hand-Over-Direct-v0 --video_interval $vid_interval --num_envs $num_envs --save_interval 10 --log_interval 10 --algorithm $algo  --headless --num_env_steps $num_runs
python ./source/standalone/workflows/harl/train.py  --task Isaac-Anymal-H1-Ball-Direct-v0 --video_interval $vid_interval --num_envs $num_envs --save_interval 10 --log_interval 10 --algorithm $algo  --headless --num_env_steps $num_runs
python ./source/standalone/workflows/harl/train.py  --task Isaac-Anymal-H1-Surf-Flat-Direct --video_interval $vid_interval --num_envs $num_envs --save_interval 10 --log_interval 10 --algorithm $algo  --headless --num_env_steps $num_runs
