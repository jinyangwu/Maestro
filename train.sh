set -x

# Activate virtual environment
# source .venv/bin/activate

# GPU settings - use GPU 0,1,2,3
export CUDA_VISIBLE_DEVICES=2,4,6,7
export RAY_memory_monitor_refresh_ms=250
export RAY_memory_usage_threshold=0.98
export RAY_DEBUG_INSTALL_LOCATION_THRESHOLD=0.99

export RAY_DISABLE_MEMORY_MONITOR=1
export RAY_DISABLE_DISK_MONITOR=1
export RAY_record_ref_creation_sites=0


# export NCCL_PXN_DISABLE=1
export NCCL_P2P_DISABLE=1


train_data=data/train_data_example.parquet
val_data=data/val_data_example.parquet
model_name=${MODEL_NAME:-Qwen3-VL-4B-Thinking}
rl_alg=grpo # gae(ppo) or grpo, if grpo, then better set n>1 otherwise the group norm can not be effective
n_gpus_per_node=4  # Using 6 GPUs: 0,1,2,3,4,5
n_nodes=1
n=8
max_concurrent_trajectories=64
batch_size=1
ppo_mini_batch_size=1
max_prompt_length=12288  # 增加以处理包含图像tokens的累积上下文
max_response_length=4096  # 减少响应长度以节省空间  
max_obs_length=1024
temperature=1.0
top_p=1.0
enable_agent=True # enable agent for tool use
strategy="fsdp"
action_stop_tokens='</search>,</answer>'
max_turns=4  # 减少轮数以避免上下文过长
kl_loss_coef=0.0
kl_coef=0
entropy_coeff=0
kl_loss_type=low_var_kl
lr=1e-6
reward_manager=torl
wandb_project=Maetro  # wandb project name, change this to your desired project name
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=8
tensor_model_parallel_size=1
gpu_memory_utilization=0.6 # Lower value to leave memory for training
do_offload=False # Enable offload to save GPU memory
use_dynamic_bsz=True # faster
ulysses_sequence_parallel_size=1 # set to 1 for normal verl behavior, otherwise it will cause OOM
fsdp_size=-1
additional_eos_token_ids=[151645] # <|im_end|> token id
mask_observations=True # mask observations for kl loss and gradient descent
enable_mtrl=False # enable multi-turn training
max_action_length=8192
model_pretty_name=$(echo $model_name | tr '/' '_' | tr '[:upper:]' '[:lower:]')
run_name_postfix="4gpu-v0"
if [ "$enable_agent" = "True" ]; then
    run_name="${reward_manager}-${strategy}-agent-${model_pretty_name}-${rl_alg}-n${n}-b${batch_size}-t${temperature}-lr${lr}${run_name_postfix}"
else
    run_name="${reward_manager}-${strategy}-${model_pretty_name}-${rl_alg}-n${n}-b${batch_size}-t${temperature}-lr${lr}${run_name_postfix}"
fi
export VERL_RUN_ID=$run_name
export NCCL_DEBUG=WARN
export VLLM_USE_V1=1
rollout_mode='async'

# temp file for action tokens as verl cannot pass special strs as params
mkdir -p logs tmp checkpoints verl_step_records "$RAY_TMPDIR"
action_stop_tokens_file=$(mktemp "tmp/action_stop_tokens.XXXXXX")
echo -e -n "$action_stop_tokens" | tee "$action_stop_tokens_file"
echo "action_stop_tokens_file=$action_stop_tokens_file"

host=$(hostname -i | awk '{print $1}')
port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$host:$port/get_observation
python -m verl_tool.servers.serve --host $host --port $port --tool_type "ipython_code" --workers_per_tool 8 --use_ray=True > logs/tool_server.log &
server_pid=$!

echo "Server (pid=$server_pid) started at $tool_server_url"

# ,'optimizer','extra'
PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.train_batch_size=$batch_size \
    data.val_batch_size=1 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.truncation='right' \
    reward_model.reward_manager=$reward_manager \
    reward_model.launch_reward_fn_async=True \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.checkpoint.save_contents=['model'] \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.strategy=$strategy \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.actor.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.agent.enable_agent=$enable_agent \
    actor_rollout_ref.agent.tool_server_url=$tool_server_url \
    actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
    actor_rollout_ref.agent.max_response_length=$max_response_length \
    actor_rollout_ref.agent.max_start_length=$max_prompt_length \
    actor_rollout_ref.agent.max_obs_length=$max_obs_length \
    actor_rollout_ref.agent.max_turns=$max_turns \
    actor_rollout_ref.agent.max_concurrent_trajectories=$max_concurrent_trajectories \
    actor_rollout_ref.agent.additional_eos_token_ids=$additional_eos_token_ids \
    actor_rollout_ref.agent.mask_observations=$mask_observations \
    actor_rollout_ref.agent.action_stop_tokens=$action_stop_tokens_file \
    actor_rollout_ref.agent.enable_mtrl=$enable_mtrl \
    actor_rollout_ref.agent.max_action_length=$max_action_length \
    +actor_rollout_ref.agent.retokenization=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=$temperature \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    critic.optim.lr=1e-5 \
    critic.strategy=$strategy \
    critic.model.path=$model_name \
    critic.model.fsdp_config.fsdp_size=$fsdp_size \
    critic.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    critic.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$wandb_project \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=checkpoints/$run_name \
    trainer.rollout_data_dir=verl_step_records/$run_name \
    trainer.validation_data_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    +trainer.remove_previous_ckpt_in_save=False \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=100 \
    trainer.total_training_steps=380


pkill -P -9 $server_pid
kill -9 $kill $server_pid
