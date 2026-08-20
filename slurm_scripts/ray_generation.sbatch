#!/bin/bash
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --job-name=daft-qwen38-27b

set -euo pipefail

export WANDB_MODE=offline

VENV=/e/project1/reformo/enrico/qwen-sglang/.venv
PY=$VENV/bin/python
RAY=$VENV/bin/ray
SCRIPT=/e/project1/reformo/enrico/qwen-sglang/main.py

export NCCL_SOCKET_IFNAME=ib0 GLOO_SOCKET_IFNAME=ib0
export NCCL_SOCKET_FAMILY=AF_INET GLOO_SOCKET_FAMILY=AF_INET

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node=${nodes[0]}
head_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" bash -c \
  "ip -o -4 addr show ib0 | awk '{print \$4}' | cut -d/ -f1")
port=6379
export RAY_ADDRESS="$head_ip:$port"
export TOTAL_GPUS=$((SLURM_JOB_NUM_NODES * 4))

srun --nodes=1 --ntasks=1 -w "$head_node" \
  $RAY start --head --node-ip-address="$head_ip" --port=$port \
  --num-cpus "$SLURM_CPUS_PER_TASK" --num-gpus 4 --block &
sleep 10

for ((i = 1; i < SLURM_JOB_NUM_NODES; i++)); do
  srun --nodes=1 --ntasks=1 -w "${nodes[$i]}" bash -c "
    $RAY start --address=$RAY_ADDRESS \
      --node-ip-address=\$(ip -o -4 addr show ib0 | awk '{print \$4}' | cut -d/ -f1) \
      --num-cpus $SLURM_CPUS_PER_TASK --num-gpus 4 --block" &
  sleep 5
done

$PY - <<'EOF'
import os, time, ray
ray.init(address=os.environ["RAY_ADDRESS"])
want, deadline = float(os.environ["TOTAL_GPUS"]), time.time() + 300
while ray.cluster_resources().get("GPU", 0) < want:
    if time.time() > deadline: raise RuntimeError("Ray cluster incomplete")
    time.sleep(5)
print("cluster ready:", ray.cluster_resources())
ray.shutdown()
EOF

srun --overlap --nodes=1 --ntasks=1 -w "$head_node" $PY "$SCRIPT" \
  --input "$1" --output "$2" "${@:3}"