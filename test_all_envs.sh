eval "$(conda shell.bash hook)"
conda activate env_isaacsim

num_runs=100_000


# python source/standalone/workflows/harl/train.py  --task Isaac-Multi-Agent-Flat-Anymal-C-Direct-v0 --video_interval 10_000 --num_envs 16 --save_interval 10 --log_interval 10 --algorithm happo --headless --num_env_steps $num_runs
# python source/standalone/workflows/harl/train.py  --task Isaac-Anymal-H1-Push-Direct-v0 --video_interval 10_000 --num_envs 16 --save_interval 10 --log_interval 10 --algorithm mappo --headless --num_env_steps $num_runs
# python source/standalone/workflows/harl/train.py  --task Isaac-Anymal-H1-Piano-Direct-v0 --video_interval 10_000 --num_envs 16 --save_interval 10 --log_interval 10 --algorithm mappo --headless --num_env_steps $num_runs
# python source/standalone/workflows/harl/train.py  --task Isaac-Shadow-Hand-Over-Direct-v0 --video_interval 10_000 --num_envs 16 --save_interval 10 --log_interval 10 --algorithm mappo --headless --num_env_steps $num_runs
# python source/standalone/workflows/harl/train.py  --task Isaac-Anymal-H1-Ball-Direct-v0 --video_interval 10_000 --num_envs 16 --save_interval 10 --log_interval 10 --algorithm mappo --headless --num_env_steps $num_runs
python source/standalone/workflows/harl/train.py  --task Isaac-Anymal-H1-Flat-Direct-v0 --video_interval 10_000 --num_envs 16 --save_interval 10 --log_interval 10 --algorithm mappo --headless --num_env_steps $num_runs
