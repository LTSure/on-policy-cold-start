import re
import datasets
from transformers import AutoTokenizer
import json
from tqdm import tqdm

content = """<think>\nOkay, let me try to figure out this problem. So, a construction company was building a tunnel. When 1/3 of the tunnel was completed at the original speed, they started using new equipment. This new equipment increased their construction speed by 20% and reduced their working hours to 80% of the original. The total time taken with the new equipment was 185 days. We need to find out how many days it would have taken if they hadn't used the new equipment and continued at the original speed.\n\nHmm. Let me break this down. Let's denote the total length of the tunnel as 1 unit for simplicity. So, they completed 1/3 of the tunnel at the original speed. Then, the remaining 2/3 was completed with the new equipment, which made them 20% faster and working 80% of the original hours per day.\n\nWait, the problem says \"reduced the working hours to 80% of the original.\" So, their daily working hours are now 80% of what they used to be. But their speed increased by 20%. So, the rate at which they construct the tunnel per day is affected both by the speed increase and the reduction in working hours.\n\nFirst, let's define some variables. Let's let the original construction speed be \"v\" units per day. Wait, but actually, maybe it's better to think in terms of time. Let me think. Suppose the original rate is such that without any changes, they would take T days to complete the tunnel. So, total work is 1 tunnel, original time is T days. Therefore, original rate is 1/T tunnels per day.\n\nBut when they started, they completed 1/3 of the tunnel at the original speed, then switched to the new equipment for the remaining 2/3. The total time taken was 185 days. We need to find T.\n\nWait, but maybe we need to relate the original time to the actual time taken with the changes. Let me approach step by step.\n\nFirst, original total time would be T days. But they didn't complete the tunnel in T days; instead, they changed their equipment partway. Let's think about the two phases: Phase 1 is building 1/3 of the tunnel at original speed, and Phase 2 is building 2/3 at the new speed with reduced working hours.\n\nLet me define the original rate as R (tunnel per day). Then, the original time to complete the whole tunnel would be T = 1/R days.\n\nIn Phase 1, they built 1/3 of the tunnel at rate R. So, the time taken for Phase 1 is (1/3)/R = 1/(3R) days.\n\nIn Phase 2, they have 2/3 of the tunnel left. Their speed increases by 20%, so the new speed is R + 20% of R = 1.2R. However, their working hours are reduced to 80% of the original. If the original working hours per day are H, then the new working hours are 0.8H. But how does this affect the construction rate?\n\nWait, speed is in tunnel per day, which is dependent on how much they work each day. If their speed was originally R tunnels per day with H hours per day, then their rate per hour is R/H tunnels per hour. When they increase their speed by 20%, that would be 1.2*(R/H) tunnels per hour. But they also reduce their working hours to 0.8H. Therefore, the new daily rate is 1.2*(R/H) * 0.8H = 1.2*0.8*R = 0.96R tunnels per day. Wait, is that correct?\n\nWait, maybe. Let me clarify. The original rate is R tunnels per day, which is based on working H hours per day. So, per hour, they complete R/H tunnels. If their speed (efficiency) increases by 20%, then their per hour rate becomes 1.2*(R/H). However, they are now working only 0.8H hours per day, so the new daily rate is 1.2*(R/H) * 0.8H = 1.2*0.8*R = 0.96R. So, their daily rate actually decreases to 0.96R? Wait, that seems counterintuitive. If they work faster but fewer hours, the net effect is a decrease in daily rate? Hmm.\n\nWait, but 20% faster in speed, but 20% less time working. Let me check the math. If original rate is R per day (H hours). Then per hour rate is R/H. After 20% increase, per hour rate is 1.2R/H. Then, working 0.8H hours per day, so daily rate is 1.2R/H * 0.8H = 0.96R. So yes, the daily rate becomes 0.96R. So even though they are working faster, the reduction in hours leads to a lower daily rate. Therefore, the time taken for Phase 2 is (2/3)/0.96R days.\n\nBut total time is 185 days, so:\n\nTime for Phase 1 + Time for Phase 2 = 185\n\nWhich is:\n\n1/(3R) + (2/3)/(0.96R) = 185\n\nWe can solve for R, then T = 1/R is the original total time.\n\nLet me compute that.\n\nFirst, let's write the equation:\n\n1/(3R) + (2/3)/(0.96R) = 185\n\nCombine the terms:\n\n[1/3 + (2/3)/0.96] / R = 185\n\nCalculate the numerator:\n\n1/3 + (2/3)/0.96\n\nFirst compute (2/3)/0.96:\n\n2/3 divided by 0.96 = (2/3)*(100/96) = (2/3)*(25/24) = (50/72) = 25/36 ≈ 0.6944\n\nThen add 1/3: 1/3 ≈ 0.3333, so total ≈ 0.6944 + 0.3333 ≈ 1.0277\n\nSo numerator ≈ 1.0277\n\nTherefore, 1.0277 / R = 185\n\nSo R ≈ 1.0277 / 185 ≈ 0.005555 tunnels per day\n\nTherefore, original time T = 1/R ≈ 1 / 0.005555 ≈ 180 days\n\nWait, so is T 180 days? Let me check the exact calculation instead of approximating.\n\nLet me compute 1/3 + (2/3)/0.96 exactly.\n\nFirst, (2/3)/0.96:\n\n0.96 is 24/25, so dividing by 24/25 is multiplying by 25/24.\n\nThus, (2/3)*(25/24) = (50/72) = 25/36.\n\nThen 1/3 is 12/36, so 12/36 + 25/36 = 37/36.\n\nTherefore, numerator is 37/36.\n\nSo the equation is (37/36)/R = 185\n\nHence, R = 37/(36*185) = 37/(6660) = 1/180 per day.\n\nTherefore, T = 1/R = 180 days.\n\nSo the original time would be 180 days.\n\nWait, so the answer is 180 days? Let me confirm.\n\nIf original time is 180 days, then original rate is 1/180 per day.\n\nPhase 1: 1/3 of the tunnel at 1/180 per day. Time = (1/3)/(1/180) = 60 days.\n\nPhase 2: 2/3 of the tunnel at the new rate. The new rate is 0.96*(1/180) = 0.96/180 = 0.005333... per day.\n\nTime for Phase 2: (2/3)/0.005333... = (2/3)/(0.005333...) ≈ (2/3)/ (8/1500) ) Because 0.005333 is 8/1500.\n\nWait, 0.005333... is 8/1500? Let me check. 8 divided by 1500 is 0.005333..., yes.\n\nSo (2/3) divided by (8/1500) = (2/3)*(1500/8) = (3000/24) = 125 days.\n\nWait, 60 days + 125 days = 185 days, which matches the given total time. Therefore, the original time T is indeed 180 days. So the answer is 180.\n\nTherefore, if they had not used the new equipment, it would have taken 180 days.\n\n**Final Answer**\n\\boxed{180}\n</think>\n\nLet \\( T \\) be the original number of days required to complete the tunnel. The original construction speed is \\( \\frac{1}{T} \\) tunnels per day. \n\nWhen \\( \\frac{1}{3} \\) of the tunnel was completed at the original speed, the time taken for this phase is:\n\\[\n\\frac{1/3}{1/T} = \\frac{T}{3} \\text{ days}\n\\]\n\nFor the remaining \\( \\frac{2}{3} \\) of the tunnel, the construction speed increased by 20% and the working hours were reduced to 80% of the original. The new construction speed is \\( 1.2 \\times \\frac{1}{T} \\) per hour, but due to reduced working hours, the effective daily rate becomes:\n\\[\n1.2 \\times \\frac{1}{T} \\times 0.8 = \\frac{0.96}{T} \\text{ tunnels per day}\n\\]\n\nThe time required to complete the remaining \\( \\frac{2}{3} \\) of the tunnel at the new rate is:\n\\[\n\\frac{2/3}{0.96/T} = \\frac{2T}{3 \\times 0.96} = \\frac{2T}{2.88} = \\frac{25T}{36} \\text{ days}\n\\]\n\nThe total time taken with the new equipment is 185 days:\n\\[\n\\frac{T}{3} + \\frac{25T}{36} = 185\n\\]\n\nCombining the terms:\n\\[\n\\frac{12T}{36} + \\frac{25T}{36} = \\frac{37T}{36} = 185\n\\]\n\nSolving for \\( T \\):\n\\[\nT = 185 \\times \\frac{36}{37} = 180\n\\]\n\nThus, if they had not used the new equipment and continued at the original speed, it would have taken \\(\\boxed{180}\\) days to complete the tunnel.
"""

data_path = "/cpfs04/user/liutianshuo/dataset/openr1"
# model_path = "/cpfs04/user/liutianshuo/model/Qwen2.5_Math_1.5B"
model_path = "/cpfs04/user/liutianshuo/model/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# output_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/train/data/openr1_data_processed_with_qwen_prompt.json"
output_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/train/data/openr1_data_processed_with_llama_3.1_prompt.json"

dataset = datasets.load_dataset(data_path, split="train")
pattern = r"</think>\s*\n\s*\n(.*)"

json_dataset = []
total_num = 0
num = 0
data_keys = ["input", "answer", "gt_answer", "subject", "level", "question", "ground_truth_answer", "target", "response"]
for data in tqdm(dataset):
    contents = data["generations"]
    for content in contents:
        format_data = {k:0 for k in data_keys}
        format_data["question"] = data["problem"]
        format_data["answer"] = data["answer"]
        format_data["gt_answer"] = data["answer"]
        format_data["ground_truth_answer"] = data["answer"]
        format_data["target"] = data["answer"]
        
        responses = re.search(pattern, content, re.DOTALL)
        if responses:
            having_content = True
            format_data["response"] = responses[0][len("</think>\n\n"):]
        else:
            having_content = False
            num += 1
        total_num += 1

        format_input = [{
            "role": "user",
            "content": data["problem"],
        }]
        format_data["input"] = tokenizer.apply_chat_template(format_input, add_generation_prompt=True, tokenize=False)
        if having_content:
            json_dataset.append(format_data)

print(total_num, num)
with open(output_path, "w") as f:
    json.dump(json_dataset, f, indent=4)