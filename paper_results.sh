eval "$(conda shell.bash hook)"

conda activate env_isaaclab

cd ./scripts/reinforcement_learning/harl

for i in {1..3}; do
    python train.py --num_envs 1000 \
    --task "Isaac-Multi-Agent-Flat-Anymal-C-Direct-v0" --seed "$i" --save_interval 10000 \
    --log_interval 1 --exp_name "multi_agent_anymal_harl" --num_env_steps 200_000_000 \
    --algorithm happo --headless
done