set -x
export VLLM_ATTENTION_BACKEND=XFORMERS
python3 -m verl.trainer.main_ppo ./scripts/toy-multistep-reasoning-v4/v4_4.yaml
