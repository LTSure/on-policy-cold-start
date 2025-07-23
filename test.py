# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # 设置模型路径
# model_dir = "/cpfs04/user/liutianshuo/verl-agent/checkpoints/verl_agent_alfworld/grpo_qwen2.5_1.5b_pposft0710/global_step_120/actor"

# # 载入 tokenizer 和模型
# tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_dir,
#     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#     device_map="auto",  # 自动放到 GPU
#     trust_remote_code=True
# )

# # 输入一个 prompt（你可以换成 ALFWorld 的 system+user 拼接）
# prompt = """You are an intelligent agent in a household. Your task is to: put the apple in the fridge.

# Thought: I see the apple on the table. The fridge is nearby.
# Action:"""

# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# # 推理生成
# with torch.no_grad():
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=100,
#         do_sample=True,
#         temperature=0.7,
#         top_p=0.9,
#         eos_token_id=tokenizer.eos_token_id,
#     )

# # 解码输出
# response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print("\n===== MODEL OUTPUT =====\n")
# print(response)






import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 原始模型结构（如 Qwen2.5-1.5B）
base_model_name = "/oss/public/user/liuts/model/Llama-3.2-3B-Instruct"

# 你的并行保存路径
actor_dir = "/cpfs04/user/liutianshuo/verl-agent/checkpoints/verl_agent_alfworld/grpo_qwen2.5_1.5b_pposft0710/global_step_100/actor"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(actor_dir, trust_remote_code=True)

# 加载基础模型结构
model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)

# 汇总模型的 sharded 权重（多 rank）
state_dict = {}
for rank in range(8):  # world size = 8
    rank_path = os.path.join(actor_dir, f"model_world_size_8_rank_{rank}.pt")
    shard = torch.load(rank_path, map_location="cpu")
    state_dict.update(shard)

# 加载合并权重
model.load_state_dict(state_dict, strict=False)

# 保存为 Hugging Face 格式
save_path = "/cpfs04/user/liutianshuo/verl-agent/checkpoints/verl_agent_alfworld/ckpt_merge"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"✅ 模型已保存为 Hugging Face 格式: {save_path}")
