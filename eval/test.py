# import json
# from evaluate import evaluate_on_policy  
# import argparse
# import numpy as np
# from tqdm import tqdm
# from pebble import ProcessPool
# from concurrent.futures import TimeoutError

# from grader import *

# from parser import *
# from utils import load_jsonl
# from python_executor import PythonExecutor


# def equal(param):
#     preds, gt = param[-2], param[-1]
#     # print(any(math_equal(pred, gt) for pred in preds))
#     # print([math_equal(pred, gt) for pred in preds])
#     # print(len(preds))
#     return math_equal(pred, gt), math_equal(pred, gt)
#     # return any(math_equal(pred, gt) for pred in preds), [math_equal(pred, gt) for pred in preds]



# def evaluate_on_policy(data_name, prompt_type, samples: list=None, file_path: str=None, max_num_samples=None, execute=False):
    # assert samples or file_path, "samples or file_path must be provided"
    # if not samples:
    #     samples = list(load_jsonl(file_path))
    # if 'idx' in samples[0]:
    #     samples = {sample['idx']: sample for sample in samples}.values()
    #     samples = sorted(samples, key=lambda x: x['idx']) 
    # else:
    #     samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]

    # if max_num_samples:
    #     print(f"max_num_samples: {max_num_samples} / {len(samples)}")
    #     samples = samples[:max_num_samples]
    
    # params = [(idx, sample["pred"], sample['gt_ans']) for idx, sample in enumerate(samples)]

    
    # scores = []
    # all_scores = []
    # timeout_cnt = 0 

    # with ProcessPool(max_workers=1) as pool:
    #     future = pool.map(equal, params, timeout=3)
    #     iterator = future.result()
    #     with tqdm(total=len(samples), desc="Evaluate") as progress_bar:
    #         for _ in range(len(samples)):
    #             try:
    #                 result = next(iterator)
    #                 # assert result[0] is True or result[0] is False
    #                 # assert len(result[1]) == 32
    #                 scores.append(result[0])
    #                 all_scores.append(result[1])
    #             except TimeoutError as error:
    #                 print("Timeout:", error)
    #                 scores.append(False)
    #                 all_scores.append([False, False, False, False])
    #                 timeout_cnt += 1
    #             except StopIteration:
    #                 break
    #             except Exception as error:
    #                 print("Error:", getattr(error, 'traceback', error))
    #                 exit()
    #             progress_bar.update(1)

    # idx = 0
    # score_mat = []

    # for sample in samples:
    #     sample['all_score'] = all_scores[idx]
    #     sample['score'] = [scores[idx]]
    #     # assert any(sample['all_score'])==sample['score'][0]
    #     # assert len(sample['all_score']) == 4
    #     # assert len(sample['score']) == 1
    #     score_mat.append(sample['score'])
    #     idx += 1

    # max_len = max([len(s) for s in score_mat])

    # for i, s in enumerate(score_mat):
    #     if len(s) < max_len:
    #         score_mat[i] = s + [s[-1]] * (max_len - len(s)) # pad

    # # output mean of each column of scores
    # col_means= np.array(score_mat).mean(axis=0)
    # mean_score = list(np.round(col_means * 100, decimals=1))

    # result_json = {
    #     "num_samples": len(samples),
    #     "num_scores": len(scores),
    #     "timeout_samples": timeout_cnt,
    #     # "empty_samples": len([s for s in samples if not s['pred'][-1]]),
    #     "empty_samples": len([s for s in samples if not s['pred'] or not s['pred'][-1]]),
    #     "acc": mean_score[0]
    # }

    # # each type score
    # if "type" in samples[0]:
    #     type_scores = {}
    #     for sample in samples:
    #         if sample['type'] not in type_scores:
    #             type_scores[sample['type']] = []
    #         type_scores[sample['type']].append(sample['score'][-1])
    #     type_scores = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in type_scores.items()}
    #     type_scores = {k: v for k, v in sorted(type_scores.items(), key=lambda item: item[0])}
    #     result_json['type_acc'] = type_scores

    # print(result_json)
    # return samples, result_json




# def evaluate_on_policy(data_name, prompt_type, samples: list=None, file_path: str=None, max_num_samples=None, execute=False):
#     assert samples or file_path, "samples or file_path must be provided"
#     if not samples:
#         samples = list(load_jsonl(file_path))
#     if 'idx' in samples[0]:
#         samples = {sample['idx']: sample for sample in samples}.values()
#         samples = sorted(samples, key=lambda x: x['idx']) 
#     else:
#         samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]

#     if max_num_samples:
#         print(f"max_num_samples: {max_num_samples} / {len(samples)}")
#         samples = samples[:max_num_samples]
    

    
#     params = [(idx, sample["pred"], sample['gt_ans']) for idx, sample in enumerate(samples)]

#     # print(f"params###############################: {params}")
#     scores = []
#     timeout_cnt = 0 


#     with ProcessPool(max_workers=1) as pool:
#         future = pool.map(math_equal_process2, params, timeout=3)
#         iterator = future.result()
#         with tqdm(total=len(samples), desc="Evaluate") as progress_bar:
#             while True:
#                 try:
#                     result = next(iterator)
#                     scores.append(result)
#                 except StopIteration:
#                     break
#                 except TimeoutError as error:
#                     print(error)
#                     scores.append(False)
#                     timeout_cnt += 1
#                 except Exception as error:
#                     print(error.traceback)
#                     exit()
#                 progress_bar.update(1) 

#     idx = 0
#     score_mat = []
#     for sample in samples:
#         sample['score'] = scores[idx: idx+1]
#         assert len(sample['score']) == 1
#         score_mat.append(sample['score'])
#         idx += 1

#     max_len = max([len(s) for s in score_mat])

#     for i, s in enumerate(score_mat):
#         if len(s) < max_len:
#             score_mat[i] = s + [s[-1]] * (max_len - len(s)) # pad

#     # output mean of each column of scores
#     col_means= np.array(score_mat).mean(axis=0)
#     mean_score = list(np.round(col_means * 100, decimals=1))

#     result_json = {
#         "num_samples": len(samples),
#         "num_scores": len(scores),
#         "timeout_samples": timeout_cnt,
#         # "empty_samples": len([s for s in samples if not s['pred'][-1]]),
#         "empty_samples": len([s for s in samples if not s['pred'] or not s['pred'][-1]]),
#         "acc": mean_score[0]
#     }

#     # each type score
#     if "type" in samples[0]:
#         type_scores = {}
#         for sample in samples:
#             if sample['type'] not in type_scores:
#                 type_scores[sample['type']] = []
#             type_scores[sample['type']].append(sample['score'][-1])
#         type_scores = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in type_scores.items()}
#         type_scores = {k: v for k, v in sorted(type_scores.items(), key=lambda item: item[0])}
#         result_json['type_acc'] = type_scores

#     print(result_json)
#     return samples, result_json



# # file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler/ppo/2/deepscaler/merge.jsonl"  
# # file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/3/deepscaler/None_qwen25-math-cot_-1_seed42_t0.6_s40000_e40200.jsonl"
# # file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/3/deepscaler/None_qwen25-math-cot_-1_seed42_t0.6_s40200_e-1.jsonl"
# # file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler/ppo/2/deepscaler/None_qwen25-math-cot_-1_seed42_t0.6_s0_e2500.jsonl"
# # file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler/ppo/2/deepscaler/None_qwen25-math-cot_-1_seed42_t0.6_s0_e2500.jsonl"
# # file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler/ppo/2/deepscaler/None_qwen25-math-cot_-1_seed42_t0.6_s40000_e-1.jsonl"
# # file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/None_qwen25-math-cot_-1_seed42_t0.6_s7500_e10000.jsonl"
# file_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_7B/lhy_test_math500_no_temp/length_4000/aime24/test_qwen25-math-cot_-1_seed42_t0.0_s0_e-1.jsonl"

# with open(file_path, 'r', encoding='utf-8') as f:
#     results = [json.loads(line) for line in f]
# # results=results[0:10]

# results, result_json = evaluate_on_policy(
#     samples=results,
#     data_name="deepscaler",         
#     prompt_type="qwen25-math-cot",  
#     execute=True
# )

# with open("results_with_eval2.jsonl", "w") as f:
#         json.dump(result_json, f, indent=4)



# output_file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler/ppo/2/deepscaler/merge_score2.jsonl"  # 设定保存路径
# with open(output_file_path, 'w', encoding='utf-8') as f:
#     for result in results:
#         f.write(json.dumps(result, ensure_ascii=False) + '\n')













# # import json

# # file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler/ppo/2/deepscaler/merge_score.jsonl"  

# # with open(file_path, 'r', encoding='utf-8') as f:
# #     for i, line in enumerate(f):

       
# #         data = json.loads(line)
# #         score=data['score']
# #         all_score=data['all_score']
# #         if any(all_score)!=score[0]:
# #             print(i)
        
      



















# -----------------merge--------------------
# import os
# import json
# import re
                  
# filenames = [
#     'None_qwen25-math-cot_-1_seed42_t0.6_s0_e2500.jsonl',
#     'None_qwen25-math-cot_-1_seed42_t0.6_s2500_e5000.jsonl',
#     'None_qwen25-math-cot_-1_seed42_t0.6_s5000_e7500.jsonl',
#     'None_qwen25-math-cot_-1_seed42_t0.6_s7500_e10000.jsonl',
#     'None_qwen25-math-cot_-1_seed42_t0.6_s10000_e12500.jsonl',
#     'None_qwen25-math-cot_-1_seed42_t0.6_s12500_e15000.jsonl',
#     'None_qwen25-math-cot_-1_seed42_t0.6_s15000_e17500.jsonl',
#     'None_qwen25-math-cot_-1_seed42_t0.6_s17500_e20000.jsonl',
#     'None_qwen25-math-cot_-1_seed42_t0.6_s20000_e22500.jsonl',
#     'None_qwen25-math-cot_-1_seed42_t0.6_s22500_e25000.jsonl',
#     'None_qwen25-math-cot_-1_seed42_t0.6_s25000_e27500.jsonl',
#     'None_qwen25-math-cot_-1_seed42_t0.6_s27500_e30000.jsonl',
#     'None_qwen25-math-cot_-1_seed42_t0.6_s30000_e32500.jsonl',
#     'None_qwen25-math-cot_-1_seed42_t0.6_s32500_e35000.jsonl',
#     'None_qwen25-math-cot_-1_seed42_t0.6_s35000_e37500.jsonl',
#     'None_qwen25-math-cot_-1_seed42_t0.6_s37500_e40000.jsonl',
#     'None_qwen25-math-cot_-1_seed42_t0.6_s40000_e-1.jsonl',
# ]

# import os

# input_folder = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0602/ppo/2/deepscaler"
# output_file  = os.path.join(input_folder, "merge.jsonl")

# filenames = [
#     'None_qwen25-math-cot_-1_seed42_t0.6_s0_e2500.jsonl',
#     'None_qwen25-math-cot_-1_seed42_t0.6_s2500_e5000.jsonl',
#     'None_qwen25-math-cot_-1_seed42_t0.6_s5000_e7500.jsonl',
#     'None_qwen25-math-cot_-1_seed42_t0.6_s7500_e10000.jsonl',
#     'None_qwen25-math-cot_-1_seed42_t0.6_s10000_e12500.jsonl',
#     'None_qwen25-math-cot_-1_seed42_t0.6_s12500_e15000.jsonl',
#     'None_qwen25-math-cot_-1_seed42_t0.6_s15000_e17500.jsonl',
#     'None_qwen25-math-cot_-1_seed42_t0.6_s17500_e20000.jsonl',
#     'None_qwen25-math-cot_-1_seed42_t0.6_s20000_e22500.jsonl',
#     'None_qwen25-math-cot_-1_seed42_t0.6_s22500_e25000.jsonl',
#     'None_qwen25-math-cot_-1_seed42_t0.6_s25000_e27500.jsonl',
#     'None_qwen25-math-cot_-1_seed42_t0.6_s27500_e30000.jsonl',
#     'None_qwen25-math-cot_-1_seed42_t0.6_s30000_e32500.jsonl',
#     'None_qwen25-math-cot_-1_seed42_t0.6_s32500_e35000.jsonl',
#     'None_qwen25-math-cot_-1_seed42_t0.6_s35000_e37500.jsonl',
#     'None_qwen25-math-cot_-1_seed42_t0.6_s37500_e40000.jsonl',
#     'None_qwen25-math-cot_-1_seed42_t0.6_s40000_e-1.jsonl',
# ]

# def count_jsonl_lines(filepath):
#     with open(filepath, 'r', encoding='utf-8') as f:
#         return sum(1 for _ in f)

# for fname in filenames:
#     path = os.path.join(input_folder, fname)
#     num_lines = count_jsonl_lines(path)
#     print(f"{fname} {num_lines} 行。")
       

# with open(output_file, 'w', encoding='utf-8') as fout:
#     for fname in filenames:
#         path = os.path.join(input_folder, fname)
#         print(f"Processing {path}")
#         with open(path, 'r', encoding='utf-8') as fin:
#             for line in fin:
#                 fout.write(line)

# print("✅ 合并完成")


# def count_lines(file_path):
#     count = 0
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for _ in f:
#             count += 1
#     return count

# # 替换为你的文件路径
# # file_path = "/oss/public/user/liuts/qwen3_30b_a3b_deepscaler/ppo/2/deepscaler/merge.jsonl"
# file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler/ppo/2/deepscaler/merge.jsonl"
# total_lines = count_lines(file_path)
# print(f"Total lines: {total_lines}")










# import json
# import random
# jsonl_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler/ppo/2/deepscaler/merge_score.jsonl"  


# def build_chat_template(user_prompt: str) -> str:
#     system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
#     system_prompt = system_prompt.replace('\n', '\\n')  # 转义换行
#     user_prompt = user_prompt.replace('\n', '\\n') 
#     return (
#         f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
#         f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
#         "<|im_start|>assistant\n"
#     )



# results=[]
# with open(jsonl_path, "r", encoding="utf-8") as f:
#     for line in f:
#         data = json.loads(line.strip())
        
#         all_score = data.get("all_score")
#         score = data.get("score")

#         if score[0]==False:
#             continue
        
#         input=data["input"]
#         # chat_tem_input=build_chat_template(input)
#         truth_answer = data["gt_ans"]
#         all_score=data['all_score']

#         true_indices = [i for i, score in enumerate(all_score) if score]
#         assert len(true_indices) > 0
#         selected_index = random.choice(true_indices)
#         response = data['response'][selected_index]
#         target = data['pred'][selected_index]

#         results.append({
#                 "input": input,
#                 "answer": target,
#                 "gt_answer": truth_answer,
#                 "subject": 0,
#                 "level": 0,
#                 "question":input,
#                 "ground_truth_answer": truth_answer,
#                 "target": target,
#                 "response": response
#             })




# output_file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler/ppo/2/deepscaler/merge_no_templete.json" 
# with open(output_file_path, 'w', encoding='utf-8') as f:
#     # for result in results:
#     #     f.write(json.dumps(result, ensure_ascii=False) + '\n')
#     json.dump(results, f, indent=4, ensure_ascii=False)







import json
import random
# jsonl_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/merge.jsonl"  

# jsonl_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0602/ppo/2/deepscaler/merge.jsonl"



# results=[]
# with open(jsonl_path, "r", encoding="utf-8") as f:
#     for line in f: 
#         data = json.loads(line.strip())

#         idx = [i for i in range(len(data['pred'])) if data['pred'][i] == data['gt_ans']]

#         if len(idx) == 0:
#             continue

#         valid_idx = [i for i in idx if len(data['response'][i]) <= 7000]

#         if len(valid_idx) == 0:
#             continue

#         # 选择最长的 response
#         # selected_index = max(valid_idx, key=lambda i: len(data['response'][i]))
#         # selected_index = min(valid_idx, key=lambda i: len(data['response'][i]))
#         # selected_index = random.choice(valid_idx)
#         selected_index = min(valid_idx, key=lambda i: abs(len(data['response'][i]) - 3500))

#         response = data['response'][selected_index]
#         target = data['pred'][selected_index]

#         input = data["input"]
#         if len(input)>2048:
#             continue

#         truth_answer = data["gt_ans"]

#         results.append({
#             "input": input,
#             "answer": target,
#             "gt_answer": truth_answer,
#             "subject": 0,
#             "level": 0,
#             "question": input,
#             "ground_truth_answer": truth_answer,
#             "target": target,
#             "response": response
#         })


# # output_file_path = f"/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/merge_filtered_no_templete_max.json" 
# # output_file_path = f"/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/merge_filtered_no_templete_shortest_{lens}.json" 
# output_file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0602/ppo/2/deepscaler/merge_filtered.json"

# # results = random.sample(results, min(4000, len(results)))

# with open(output_file_path, 'w', encoding='utf-8') as f:
#     # for result in results:
#     #     f.write(json.dumps(result, ensure_ascii=False) + '\n')
#     json.dump(results, f, indent=4, ensure_ascii=False)







import json


# file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_7B_sft_from_base_deepscaler——0513/lts_test/aime24/test_qwen25-math-cot_-1_seed42_t0.7_s0_e30.jsonl"
# file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5-Math-7B_ppo_from_base_deepscaler——0513/lts_test/aime24/test_qwen25-math-cot_-1_seed42_t0.7_s0_e30.jsonl"
# file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_7B/lts_test/aime24/test_qwen25-math-cot_-1_seed42_t0.7_s0_e30.jsonl"
# file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/3/deepscaler/None_qwen25-math-cot_-1_seed42_t0.6_s40000_e40200.jsonl"


# file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/None_qwen25-math-cot_-1_seed42_t0.6_s7500_e10000.jsonl"
# file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/None_qwen25-math-cot_-1_seed42_t0.6_s0_e2500.jsonl"
# file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/None_qwen25-math-cot_-1_seed42_t0.6_s2500_e5000.jsonl"
# file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/None_qwen25-math-cot_-1_seed42_t0.6_s5000_e7500.jsonl"


# file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/None_qwen25-math-cot_-1_seed42_t0.6_s17500_e20000.jsonl"
# file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/None_qwen25-math-cot_-1_seed42_t0.6_s10000_e12500.jsonl"
# file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/None_qwen25-math-cot_-1_seed42_t0.6_s12500_e15000.jsonl"
# file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/None_qwen25-math-cot_-1_seed42_t0.6_s15000_e17500.jsonl"


# file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/None_qwen25-math-cot_-1_seed42_t0.6_s27500_e30000.jsonl"
# file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/None_qwen25-math-cot_-1_seed42_t0.6_s20000_e22500.jsonl"
# file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/None_qwen25-math-cot_-1_seed42_t0.6_s22500_e25000.jsonl"
# file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/None_qwen25-math-cot_-1_seed42_t0.6_s25000_e27500.jsonl"


# file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/None_qwen25-math-cot_-1_seed42_t0.6_s37500_e40000.jsonl"
# file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/None_qwen25-math-cot_-1_seed42_t0.6_s30000_e32500.jsonl"
# file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/None_qwen25-math-cot_-1_seed42_t0.6_s32500_e35000.jsonl"
# file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/None_qwen25-math-cot_-1_seed42_t0.6_s35000_e37500.jsonl"

# file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/merge-0-10000.jsonl"
# file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler——0516/aime24/test_qwen25-math-cot_-1_seed42_t0.7_s0_e-1.jsonl"
# file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/merge_filtered_no_templete_shortest.json"
# file_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off16——0526/_actor/3/lhy_test_math500_no_temp/length_6000/MATH-500/test_qwen25-math-cot_-1_seed42_t0.0_s0_e-1.jsonl"

# file_path="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_7B/lhy_test_math500_no_temp/length_7000/aime24/test_qwen25-math-cot_-1_seed42_t0.0_s0_e-1.jsonl"




# p=[]
# a=[]
# with open(file_path, 'r', encoding='utf-8') as f:
#     for line_num, line in enumerate(f, 1):
#         try:
#             data = json.loads(line)
#             response = data.get("response", "")
#             if len(data['response'])>3396 or len(data['input'])>700:
#                 print(line_num)
          
           
#             p.append(len(data["response"]))
#             a.append(data['input'])
#         except json.JSONDecodeError as e:
#             print(f"Line {line_num}: JSON decode error - {e}")

# print(sum(p)/len(p), max(p))
# print(len(p),len(a), max(a))






import json


# file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0602/ppo/2/deepscaler/merge_0_2w_filtered_4k.json"
# file_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0602/ppo/2/deepscaler/merge_filtered.json"

# p=[]
# a=[]

# with open(file_path, "r", encoding="utf-8") as f:
#     datas = json.load(f)


# for data in datas:
#     try:
#         response = data.get("response", "")
#         if len(data['response'])>7000 or len(data['input'])>2048:
#             print(line_num)
        
#         p.append(len(data["response"]))
#         a.append(len(data['input']))
#     except json.JSONDecodeError as e:
#         print(f"Line {line_num}: JSON decode error - {e}")

# print(sum(p)/len(p),max(p))
# print(sum(a)/len(a),max(a))





import json

# file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler——0516/aime24/test_qwen25-math-cot_-1_seed42_t0.7_s0_e-1.jsonl"

# file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/merge_filtered_no_templete_shortest.json"
# file_path="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0602/ppo/2/deepscaler/None_qwen25-math-cot_-1_seed42_t0.6_s0_e2500.jsonl"

# response_lengths = []

# with open(file_path, 'r', encoding='utf-8') as f:
#     for line_num, line in enumerate(f, 1):
#         try:
#             data = json.loads(line)
#             # response = data.get("response", "")
#             response = data.get("response", "")
#             response_length = [len(r) for r in response] 
#             # response_length = len(response)
#             # response_lengths.append(response_length)
#             response_lengths.extend(response_length)
#             # print(f"Line {line_num}: length = {response_length}")
#         except json.JSONDecodeError as e:
#             print(f"Line {line_num}: JSON decode error - {e}")

# # 可选：统计平均长度
# if response_lengths:
#     avg_length = sum(response_lengths) / len(response_lengths)
#     print(f"\nAverage response length: {avg_length:.2f}")



# file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler/ppo/2/deepscaler/merge_new.json"
# file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/train/data/openr1_data_processed_with_qwen_prompt_filter.json"
# file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/merge-0-10000_filtered_no_templete_shortest.json"
# file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/merge_filtered_no_templete_shortest.json"

# file_path = f"/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/merge_filtered_no_templete_max.json" 
# file_path = f"/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/merge_filtered_no_templete_shortest.json"
# file_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/merge_filtered_no_templete_shortest_7000.json" 
# file_path="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0602/ppo/2/deepscaler/merge_filtered.json"

# with open(file_path, "r", encoding="utf-8") as f:
#     data = json.load(f)

# lengths = [len(item["response"]) for item in data if "response" in item]
# lengths = [len(item["response"]) for item in data]
# lengths = [len(item["input"]) for item in data]
# lengths = [len(item["input"])+len(item["response"])  for item in data]
# print(len(lengths))

# for i, l in enumerate(lengths):
#     print(f"Item {i}: length = {l} chars")
# print(f"\n平均长度：{sum(lengths) / len(lengths):.2f} ")
# lens=8192
# print(f"{lens}>max长度：{  len([i for i in lengths if i > lens])  } ")
# print(f"\n最长长度：{max(lengths) } ")

# a=0
# p=0
# for i in range(len(data)):
#     a+=1
#     if data[i]["target"]==data[i]["ground_truth_answer"]:
#         p+=1
# print(a,p)




























# import argparse
# import numpy as np
# from tqdm import tqdm
# from pebble import ProcessPool
# from concurrent.futures import TimeoutError
# from grader import *
# from parser import *
# from utils import load_jsonl
# from python_executor import PythonExecutor



# def evaluate_on_policy(data_name, prompt_type, samples: list=None, file_path: str=None, max_num_samples=None, execute=False):
#     assert samples or file_path, "samples or file_path must be provided"
#     if not samples:
#         samples = list(load_jsonl(file_path))
#     if 'idx' in samples[0]:
#         samples = {sample['idx']: sample for sample in samples}.values()
#         samples = sorted(samples, key=lambda x: x['idx']) 
#     else:
#         samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]

#     if max_num_samples:
#         print(f"max_num_samples: {max_num_samples} / {len(samples)}")
#         samples = samples[:max_num_samples]
    

    
#     # params = [(idx, sample["pred"], sample['gt_ans']) for idx, sample in enumerate(samples)]
#     params = [(idx, sample["target"], sample['ground_truth_answer']) for idx, sample in enumerate(samples)]

#     scores = []
#     timeout_cnt = 0 


#     with ProcessPool(max_workers=1) as pool:
#         future = pool.map(math_equal_process2, params, timeout=3)
#         iterator = future.result()
#         with tqdm(total=len(samples), desc="Evaluate") as progress_bar:
#             while True:
#                 try:
#                     result = next(iterator)
#                     scores.append(result)
#                 except StopIteration:
#                     break
#                 except TimeoutError as error:
#                     print(error)
#                     scores.append(False)
#                     timeout_cnt += 1
#                 except Exception as error:
#                     print(error.traceback)
#                     exit()
#                 progress_bar.update(1) 

#     idx = 0
#     score_mat = []
#     for sample in samples:
#         sample['score'] = scores[idx: idx+1]
#         assert len(sample['score']) == 1
#         score_mat.append(sample['score'])
#         idx += 1

#     max_len = max([len(s) for s in score_mat])

#     for i, s in enumerate(score_mat):
#         if len(s) < max_len:
#             score_mat[i] = s + [s[-1]] * (max_len - len(s)) # pad

#     # output mean of each column of scores
#     col_means= np.array(score_mat).mean(axis=0)
#     mean_score = list(np.round(col_means * 100, decimals=1))

#     result_json = {
#         "num_samples": len(samples),
#         "num_scores": len(scores),
#         "timeout_samples": timeout_cnt,
#         # "empty_samples": len([s for s in samples if not s['target'][-1]]),
#         "empty_samples": len([s for s in samples if not s['target'] or not s['target'][-1]]),
#         "acc": mean_score[0]
#     }

#     # each type score
#     if "type" in samples[0]:
#         type_scores = {}
#         for sample in samples:
#             if sample['type'] not in type_scores:
#                 type_scores[sample['type']] = []
#             type_scores[sample['type']].append(sample['score'][-1])
#         type_scores = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in type_scores.items()}
#         type_scores = {k: v for k, v in sorted(type_scores.items(), key=lambda item: item[0])}
#         result_json['type_acc'] = type_scores

#     print(result_json)
#     return samples, result_json



# file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0602/ppo/2/deepscaler/merge_0_2w_filtered_4k.json"
# file_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0602/ppo/2/deepscaler/merge_filtered.json"

# with open(file_path, 'r', encoding='utf-8') as f:
#     results = json.load(f)

# results, result_json = evaluate_on_policy(
#     samples=results,
#     data_name="deepscaler",         
#     prompt_type="qwen25-math-cot",  
#     execute=True
# )



# from huggingface_hub import snapshot_download

# # snapshot_download(repo_id="HuggingFaceH4/MATH-500", repo_type="dataset", local_dir="./data/MATH-500")
# snapshot_download(repo_id="HuggingFaceH4/MATH-500", repo_type="dataset", local_dir="./data/MATH-500")




import os
import json

# base_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_max——0517/lhy_test_no_temp"
# base_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_sft_from_base_deepscaler_max——0517/lhy_test_no_temp"
# base_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B/lhy_test_no_temp"

# base_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_7B_ppo_from_base_deepscaler_max——0519/lhy_test_no_temp"
# base_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_7B_sft_from_base_deepscaler_max——0519/lhy_test_no_temp"
# base_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_7B/lhy_test_no_temp"

# base_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_max——0517/lhy_test_math500_no_temp"
# base_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_sft_from_base_deepscaler_max——0517/lhy_test_math500_no_temp"
# base_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B/lhy_test_math500_no_temp"

# base_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_7B_ppo_from_base_deepscaler_max——0519/lhy_test_math500_no_temp"
# base_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_7B_sft_from_base_deepscaler_max——0519/lhy_test_math500_no_temp"
# base_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_7B/lhy_test_math500_no_temp"

# base_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_7B_ppo_from_base_deepscaler_0_10000_shortest——0519/lhy_test_math500_no_temp"
# base_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_7B_sft_from_base_deepscaler_0_10000_shortest——0519/lhy_test_math500_no_temp"

# base_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_sft_from_base_deepscaler_shortest——0516/lhy_test_math500_no_temp"
# base_path = "/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_shortest——0516/lhy_test_math500_no_temp"

# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_max_offpolicy——0521/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_max_onpolicy——0521/lhy_test_math500_no_temp"


# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off——0522/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_on——0522/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_norm——0522/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_max_lr——0522/lhy_test_math500_no_temp"

# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_sft_from_base_deepscaler_max——0522/lhy_test_math500_no_temp"

# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off2——0522/lhy_test_math500_no_temp"
# base_path="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off3——0522/lhy_test_math500_no_temp"

# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off4——0523/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off5——0523/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off4——0523/_actor/4/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off5——0523/_actor/4/lhy_test_math500_no_temp"

# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off6——0523/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off6——0523/_actor/4/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off7——0523/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off7——0523/_actor/4/lhy_test_math500_no_temp"



# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off8——0524/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off8——0524/_actor/4/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off9——0524/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off9——0524/_actor/4/lhy_test_math500_no_temp"

# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off11——0524/_actor/2/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off11——0524/_actor/3/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off11——0524/_actor/4/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off11——0524/lhy_test_math500_no_temp"


# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off10——0524/_actor/2/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off10——0524/_actor/3/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off10——0524/_actor/4/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off10——0524/lhy_test_math500_no_temp"

# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off13——0525/_actor/2/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off13——0525/_actor/3/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off13——0525/_actor/4/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off13——0525/lhy_test_math500_no_temp"


# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off12——0525/_actor/2/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off12——0525/_actor/3/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off12——0525/_actor/4/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off12——0525/lhy_test_math500_no_temp"


# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off15——0525/_actor/2/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off15——0525/_actor/3/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off15——0525/_actor/4/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off15——0525/lhy_test_math500_no_temp"


# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off14——0525/_actor/2/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off14——0525/_actor/3/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off14——0525/_actor/4/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off14——0525/lhy_test_math500_no_temp"


# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off16——0526/_actor/2/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off16——0526/_actor/3/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off16——0526/_actor/4/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off16——0526/lhy_test_math500_no_temp"


# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off17——0526/_actor/1/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off17——0526/_actor/2/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off17——0526/_actor/3/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off17——0526/_actor/4/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off17——0526/lhy_test_math500_no_temp"



# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off19——0526/_actor/1/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off19——0526/_actor/2/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off19——0526/_actor/3/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off19——0526/_actor/4/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off19——0526/lhy_test_math500_no_temp"

# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off18——0526/_actor/1/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off18——0526/_actor/2/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off18——0526/_actor/3/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off18——0526/_actor/4/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off18——0526/lhy_test_math500_no_temp"


# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off20——0527/_actor/1/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off20——0527/_actor/2/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off20——0527/_actor/3/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off20——0527/_actor/4/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off20——0527/lhy_test_math500_no_temp"


# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off22_entropy——0527/_actor/1/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off22_entropy——0527/_actor/2/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off22_entropy——0527/_actor/3/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off22_entropy——0527/_actor/4/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off22_entropy——0527/lhy_test_math500_no_temp"

# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off21_entropy——0527/_actor/1/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off21_entropy——0527/_actor/2/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off21_entropy——0527/_actor/3/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off21_entropy——0527/_actor/4/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off21_entropy——0527/lhy_test_math500_no_temp"


# 大于77：
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off6——0523/_actor/4/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off7——0523/_actor/4/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off2——0522/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off11——0524/_actor/3/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off11——0524/_actor/4/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off10——0524/_actor/3/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off13——0525/_actor/4/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off12——0525/_actor/3/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off15——0525/_actor/3/lhy_test_math500_no_temp"
# (79.8)


# 大于78：
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off14——0525/_actor/2/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off14——0525/_actor/3/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off16——0526/_actor/3/lhy_test_math500_no_temp"
# base_path ="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off20——0527/_actor/2/lhy_test_math500_no_temp"



# base_path="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_1.5B/lhy_test_math500_no_temp"
# base_path="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/eval_results/Qwen2.5_Math_7B/lhy_test_math500_no_temp"


# data_set="aime24"
# data_set="MATH-500"

# length_range = range(4000, 20001, 500)
# length_range = range(6000, 12000, 500)
# results = []

# for length in length_range:
#     json_path = os.path.join(
#         base_path,
#         f"length_{length}",
#         f"{data_set}",
#         "test_qwen25-math-cot_-1_seed42_t0.0_s0_e-1_metrics.json"
#     )


#     try:
#         with open(json_path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#             acc = data.get("acc", None)
#             if acc is not None:
#                 results.append({"length": length, "acc": acc})
#                 print(f"✅ Loaded: length={length}, acc={acc}")
#             else:
#                 print(f"⚠️ 'acc' not found in file: {json_path}")
#     except FileNotFoundError:
#         print(f"❌ File not found: {json_path}")
#     except json.JSONDecodeError:
#         print(f"❌ JSON decode error: {json_path}")

# # 输出到屏幕
# print(f"\n{base_path} ")
# a=[]
# for item in results:
#     # if item['acc']<73:
#     #     continue
#     a.append(item['acc'])
#     print(f"Length: {item['length']}, Accuracy: {item['acc']}")


# print(sum(a)/len(a))

# # 可选：保存为 JSON 文件
# output_file = "length_acc_summary.json"
# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(results, f, ensure_ascii=False, indent=2)
#     print(f"\n📁 Saved summary to {output_file}")





import json

# def convert_to_chat_format(data):
    # sys_prompt="You are a program verification assistant. \nBelow is Dafny code with hints removed (hints_removed). \nPlease generate the complete and correct Dafny program (ground_truth) based on it."

    # chat= {
    #     "input": data["conversations"][0]["content"],
    #     "response": data["conversations"][1]["content"],
    #     "target":"",
    #     "ground_truth_answer":"",
    # }

#     chat= {
#         "input": data["input"],
#         "response": data["response"],
#         "target":"",
#         "ground_truth_answer":"",
#     }
#     return chat

# def process_json(input_path, output_path):
#     with open(input_path, 'r', encoding='utf-8') as f:
#         dataset = json.load(f)

#     processed = []
#     for item in dataset:
#         if len(item["input"])<4500 and len(item["response"])<4500:
#             chat_format = convert_to_chat_format(item)
#             processed.append(chat_format)

#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(processed, f, indent=2, ensure_ascii=False)
    
#     print(len(processed))

# if __name__ == "__main__":
#     # input_json = "/cpfs04/user/liutianshuo/math/simpleRL-reason/train/data/alfworld-gpt4-45k_2.json"  # 你的输入json文件路径
#     # output_json = "/cpfs04/user/liutianshuo/math/simpleRL-reason/train/data/alfworld-gpt4-45k_3.json"  # 输出转换后json文件路径
#     input_json = "/cpfs04/user/liutianshuo/math/simpleRL-reason/train/data/dafnybench_test3.json"
#     output_json = "/cpfs04/user/liutianshuo/math/simpleRL-reason/train/data/dafnybench_test4.json"
#     process_json(input_json, output_json)
#     print(f"转换完成，结果已保存到 {output_json}")



# input_json = "/cpfs04/user/liutianshuo/math/simpleRL-reason/train/data/alfworld-gpt4-45k_3.json"  # 你的输入json文件路径
# input_json = "/cpfs04/user/liutianshuo/math/simpleRL-reason/train/data/dafnybench_test3.json"
# input_json = "/cpfs04/user/liutianshuo/math/simpleRL-reason/train/data/dafnybench_test3_filter_4.5k.json"
# with open(input_json, 'r', encoding='utf-8') as f:
#         dataset = json.load(f)
# # l=[len(item['input']) for item in dataset ]
# l=[len(item['response']) for item in dataset]
# # l=[len(item['input']) for item in dataset if len(item['input'])<4500]
# # l=[len(item['response']) for item in dataset if len(item['input'])<4500]
# print(len(l),len(dataset))
# print(sum(l)/len(l),max(l))




# import json

# input_path="/cpfs04/user/liutianshuo/math/simpleRL-reason/train/data/alfworld-gpt4-45k.json"
# output_path="/cpfs04/user/liutianshuo/math/simpleRL-reason/train/data/alfworld-gpt4-45k_2.json"
# def remove_image_tag(input_path, output_path):
#     with open(input_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)

#     # 支持单条或多条数据（比如一个 list）
#     if isinstance(data, dict):
#         data = [data]

#     for entry in data:
#         if "conversations" in entry:
#             for conv in entry["conversations"]:
#                 if conv.get("from") == "human" and isinstance(conv.get("value"), str):
#                     # 去掉开头的 <image>\n 或 <image>
#                     if conv["value"].startswith("<image>\n"):
#                         conv["value"] = conv["value"].replace("<image>\n", "", 1)
#                     elif conv["value"].startswith("<image>"):
#                         conv["value"] = conv["value"].replace("<image>", "", 1)

#     # 写入新文件
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(data, f, indent=2, ensure_ascii=False)

# # 示例用法
# remove_image_tag(input_path, output_path)


# import json

# def convert_conversations_format(input_file, output_file):
#     with open(input_file, 'r', encoding='utf-8') as f:
#         data = json.load(f)

#     # 如果是一个列表，逐个处理；否则假设是一个 dict
#     if isinstance(data, list):
#         for item in data:
#             if "conversations" in item:
#                 item["conversations"] = [
#                     {
#                         "role": conv["from"],
#                         "content": conv["value"]
#                     }
#                     for conv in item["conversations"]
#                 ]
#     elif isinstance(data, dict) and "conversations" in data:
#         data["conversations"] = [
#             {
#                 "role": conv["from"],
#                 "content": conv["value"]
#             }
#             for conv in data["conversations"]
#         ]

#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(data, f, indent=2, ensure_ascii=False)

# input_path="/cpfs04/user/liutianshuo/math/simpleRL-reason/train/data/alfworld-gpt4-45k_2.json"
# output_path="/cpfs04/user/liutianshuo/math/simpleRL-reason/train/data/alfworld-gpt4-45k_3.json"
# convert_conversations_format(input_path, output_path)






# print(c1,c2,len(data))


# print(sum(prompt)/len(prompt), max(prompt))
# print(sum(response)/len(response), max(response))

import json
import re
# 输入输出文件路径

# input_file = '/cpfs04/user/liutianshuo/math/simpleRL-reason/train/data/alfworld-gpt4-45k_2.json'
# output_file = '/cpfs04/user/liutianshuo/math/simpleRL-reason/train/data/alfworld-gpt4-45k_.json'

# # 读取原始 JSON 文件
# with open(input_file, 'r', encoding='utf-8') as f:
#     dataset = json.load(f)



# def extract_action_value(text: str) -> str:
#     a=text
#     pattern = r'"action.*'  # 匹配 "action 开头直到行尾的内容
#     match = re.search(pattern, text)
#     if match:
#         result = match.group(0)  # 包含 "action 及其后面内容
#         assert len(result)!=0
#         return result
#     else:
#         print("没找到匹配")
#         return a


# for data in dataset:
#     for convo in data.get("conversations"):
#         # if convo["role"] == "human":
#         #     convo["role"] = "user"
#         # elif convo["role"] == "gpt":
#         if convo["role"] == "assistant":
#             convo["content"] = extract_action_value(convo["content"])


# with open(output_file, 'w', encoding='utf-8') as f:
#     json.dump(dataset, f, ensure_ascii=False, indent=2)


# input_file = '/cpfs04/user/liutianshuo/AgentBoard/data/alfworld/test.jsonl'
# # output_file = '/cpfs04/user/liutianshuo/math/simpleRL-reason/train/data/alfworld-gpt4-45k_.json'

# data = []
# with open(input_file, 'r', encoding='utf-8') as f:
#     for line in f:
#         if line.strip():  
#             data.append(json.loads(line))

# a=0
# for item in data:
#     difficulty=item['difficulty']
#     if difficulty=='easy':
#         a+=1
# print(a,len(data)-a)


import json

# def convert_conversations_format(json_data):
#     for sample in json_data:
#         new_conversations = []
#         for turn in sample.get("conversations", []):
#             role_map = {"human": "user", "gpt": "assistant"}
#             new_turn = {
#                 "role": role_map.get(turn["from"], turn["from"]),
#                 "content": turn["value"]
#             }
#             new_conversations.append(new_turn)
#         sample["conversations"] = new_conversations
#     return json_data

# # 文件读写示例
# input_file = "/cpfs04/user/liutianshuo/math/simpleRL-reason/train/data/alfworld_sft_mt.json"         # 替换为你的输入文件路径
# output_file = "/cpfs04/user/liutianshuo/math/simpleRL-reason/train/data/alfworld_sft_mt_2.json"   # 替换为输出文件路径

# with open(input_file, "r", encoding="utf-8") as f:
#     data = json.load(f)

# print(len(data))

# converted_data = convert_conversations_format(data)

# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(converted_data, f, ensure_ascii=False, indent=2)





# from transformers import AutoTokenizer
# import torch
# import torch.distributed as dist
# import torch.nn as nn
# import torch.nn.functional as F
# from peft import LoraConfig, TaskType

# prompt1="""
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>

# Cutting Knowledge Date: December 2023
# Today Date: 21 Jun 2025

# <|eot_id|><|start_header_id|>user<|end_header_id|>

# How are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

# I am fine.<|eot_id|><|start_header_id|>user<|end_header_id|>

# That is great.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

# You are right<|eot_id|>
# """

# prompt2="""
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>

# Cutting Knowledge Date: December 2023
# Today Date: 21 Jun 2025

# <|eot_id|><|start_header_id|>user<|end_header_id|>

# Are you ok?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

# I am OK.<|eot_id|><|start_header_id|>user<|end_header_id|>

# Nice to meet you.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

# Nice to meet you, too.<|eot_id|>
# """
# prompt=[prompt1,prompt2,]
# prompt=['<|eot_id|>',]
# #   27,     91,     68,    354,    842,     91,     29

# MODEL_NAME="Llama-3.2-3B-Instruct"
# MODEL=f"/oss/public/user/liuts/model/{MODEL_NAME}"
# tokenizer = AutoTokenizer.from_pretrained(MODEL)  # 你用的模型名
# tokenizer.padding_side = "right"

# tokenizer.pad_token = tokenizer.eos_token
# sequences = tokenizer(
#             prompt,
#             return_tensors="pt",
#             add_special_tokens=False,
#             max_length=7000,
#             padding=True,
#             truncation=True,
#         ).input_ids

# eos_token_id = tokenizer.eos_token_id
# pad_token_id = tokenizer.pad_token_id
# bos_token_id = tokenizer.bos_token_id

# print(f'eos_token_id:{eos_token_id}')
# print(f'pad_token_id:{pad_token_id}')
# print(f'bos_token_id:{bos_token_id}')
# print(sequences)
# print(tokenizer.decode([bos_token_id], skip_special_tokens=False))
# print(tokenizer.decode([eos_token_id], skip_special_tokens=False))
# print(tokenizer.decode([pad_token_id], skip_special_tokens=False))
# for i, row in enumerate(sequences):
#     text = tokenizer.decode(row, skip_special_tokens=False)
#     print(f"[Sample {i}] {text}")
# print('===================================')
# attention_mask = (sequences != eos_token_id) & (sequences != pad_token_id)
# print(attention_mask)
# attention_mask = attention_mask.long()  


# seq_length = sequences.size(1)
# eos_indices = seq_length - attention_mask.flip(dims=[1]).argmax(dim=1, keepdim=True).clamp(min=1)

# sequences = sequences.clone()  
# sequences.scatter_(dim=1, index=eos_indices, value=eos_token_id)
# print(sequences)

# first_token_indices = attention_mask.argmax(dim=1, keepdim=True)

# mask = torch.arange(seq_length).unsqueeze(0).expand(sequences.size(0), -1).to(sequences.device)

# attention_mask = ((mask >= first_token_indices) & (mask <= eos_indices)).long()
# for i, row in enumerate(sequences):
#     text = tokenizer.decode(row, skip_special_tokens=False)
#     print(f"[Sample {i}] {text}")
#     print(attention_mask[i])
# print(sequences.shape)
# print(attention_mask.shape)
# print('===================================')


# tensor([[False, False, False,  ..., False, False, False],
#         [False, False, False,  ..., False, False, False],
#         [False, False, False,  ..., False, False, False],
#         ...,
#         [False, False, False,  ..., False, False, False],
#         [False, False, False,  ..., False, False, False],
#         [False, False, False,  ..., False, False, False]], device='cuda:0')




import torch

# 模拟一个 batch=3，seq_len=10 的 log_probs（通常是模型输出 log_softmax 后的结果）
# log_probs = torch.tensor([
#     [-1.2, -2.3, -0.7, -1.5, -2.0, -3.0, -2.5, -0.9, -1.1, -2.2],  # batch 0
#     [-0.5, -1.1, -3.1, -2.2, -0.8, -1.5, -2.5, -0.6, -1.7, -2.1],  # batch 1
#     [-0.9, -1.8, -1.3, -1.2, -2.0, -2.9, -1.5, -1.1, -2.3, -2.7],  # batch 2
# ])

# # 构造一个 action_mask，只关注部分 token（例如 valid action 的位置）
# action_mask = torch.tensor([
#     [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],  # batch 0: positions 1, 2, 6
#     [0, 1, 1, 1, 1, 0, 0, 1, 1, 0],  # batch 1: positions 0, 4, 7
#     [0, 1, 1, 0, 0, 1, 1, 1, 1, 0],  # batch 2: positions 5, 6, 7
# ], dtype=torch.bool)

# # 提取被 mask 出来的有效位置的 log_probs
# valid_log_probs = log_probs[action_mask]
# print("Valid log_probs:\n", valid_log_probs)

# # 转换为原始概率（exp）
# valid_probs = torch.exp(valid_log_probs)
# print("\nValid probs:\n", valid_probs)


# import json
# from collections import defaultdict

# input_path = "/cpfs04/user/liutianshuo/human-eval/results/samples_Llama-3.2-3B-Instruct_ppo_sft_alfworld_muti_turn——0623/_actor/3_16.jsonl"
# output_path = "/cpfs04/user/liutianshuo/human-eval/results/samples_Llama-3.2-3B-Instruct_ppo_sft_alfworld_muti_turn——0623/_actor/3_16_merge.jsonl"

# # 1. 收集所有 completion
# merged_data = defaultdict(list)

# with open(input_path, "r") as f:
#     for line in f:
#         sample = json.loads(line)
#         task_id = sample["task_id"]
#         completion = sample["completion"]
#         merged_data[task_id].append(completion)

# # 2. 写入合并后的文件
# with open(output_path, "w") as f:
#     for task_id, completions in merged_data.items():
#         merged_entry = {
#             "task_id": task_id,
#             "completion": completions  # 注意是 plural
#         }
#         f.write(json.dumps(merged_entry) + "\n")

# print(f"✅ 合并完成，共处理 {len(merged_data)} 个任务。输出文件：{output_path}")


# import json

# # 输入输出路径
# input_path = "/cpfs04/user/liutianshuo/CodeEval-Pro/dataset/MbppPlus-OriginFmt.jsonl"
# output_path = "/cpfs04/user/liutianshuo/CodeEval-Pro/dataset/MbppPlus-OriginFmt.json"

# # 逐行读取 JSONL，每行是一个字典
# with open(input_path, "r", encoding="utf-8") as f:
#     data = [json.loads(line) for line in f]

# # 写入为标准 JSON 文件（列表形式）
# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(data, f, indent=2, ensure_ascii=False)

# print(f"已成功将 {input_path} 转换为 {output_path}")

# import json

# def convert_mbpp_to_evalplus_format(input_path, output_path):
#     with open(input_path, 'r') as f:
#         data = json.load(f)

#     new_data = []
#     for example in data:
#         task_id = example["task_id"]
#         raw_problem = example["text"]
#         raw_solution = example["code"]
#         new_problem = raw_problem  # 如果没变异可以直接复用
#         new_solution = raw_solution
#         test_code = example.get("test_setup_code", "") + "\n" + "\n".join(example["test_list"])

#         new_data.append({
#             "task_id": task_id,
#             "raw_problem": raw_problem,
#             "raw_solution": raw_solution,
#             "new_problem": new_problem,
#             "new_solution": new_solution,
#             "test_code": test_code
#         })

#     with open(output_path, 'w') as f:
#         json.dump(new_data, f, indent=2)




# import json
# from transformers import AutoTokenizer
# from tqdm import tqdm

# # 加载 tokenizer（根据你的模型名称或路径修改）
# tokenizer = AutoTokenizer.from_pretrained("/oss/public/user/liuts/model/Llama-3.2-3B-Instruct", use_fast=False)

# # 是否有 apply_chat_template 方法
# if not hasattr(tokenizer, "apply_chat_template"):
#     raise NotImplementedError("This tokenizer does not support apply_chat_template.")

# # 加载数据
# with open("/cpfs04/user/liutianshuo/math/simpleRL-reason/train/data/alfworld_sft_mt_2.json", "r") as f:
#     data = json.load(f)

# token_counts = []
# max_len = 0

# # 遍历数据
# for sample in tqdm(data):
#     conversations = sample["conversations"]
    
#     # 应用 chat 模板，转为字符串输入
#     prompt_str = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
    
#     # tokenize 得到 token 数量
#     token_ids = tokenizer(prompt_str, return_tensors="pt").input_ids
#     num_tokens = token_ids.shape[1]
    
#     token_counts.append(num_tokens)
#     max_len = max(max_len, num_tokens)

# # 输出统计信息
# print(f"Total samples: {len(token_counts)}")
# print(f"Average token length: {sum(token_counts) / len(token_counts):.2f}")
# print(f"Max token length: {max_len}")


# import json

# # 读取原始 JSON 文件
# with open("/cpfs04/user/liutianshuo/math/simpleRL-reason/train/data/alfworld_sft_mt_2.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# # 创建新列表，只保留每项的前三条对话
# new_data = []
# for i in range(len(data)):
#     item=data[i]
#     conversations = item.get("conversations", [])[:3]
#     new_item={
#         "answer":"0",
#         "data_source":"text",
#         "prompt":conversations,
#         "ability":"agent",
#         "extra_info":{"index":i,"split":"train"}
#     }


#     new_data.append(new_item)

# # 保存为新的 JSON 文件
# with open("/cpfs04/user/liutianshuo/verl-agent/train_data/train.json", "w", encoding="utf-8") as f:
#     json.dump(new_data, f, ensure_ascii=False, indent=2)


# import json
# import pandas as pd
# import pyarrow as pa
# import pyarrow.parquet as pq

# # 路径设定
# json_file = "/cpfs04/user/liutianshuo/verl-agent/train_data/train.json"
# parquet_file = "/cpfs04/user/liutianshuo/verl-agent/train_data/train.parquet"

# # 读取 JSON 文件
# with open(json_file, 'r', encoding='utf-8') as f:
#     data = json.load(f)

# # 转为 DataFrame（注意 JSON 是一个 list）
# df = pd.DataFrame(data)

# # 写入 Parquet 文件
# table = pa.Table.from_pandas(df)
# pq.write_table(table, parquet_file)

# print(f"✅ 成功将 {json_file} 转换为 {parquet_file}")



# from transformers import AutoTokenizer


# model_name = "/cpfs04/user/liutianshuo/math/simpleRL-reason/train/checkpoints/Qwen2.5_Math_1.5B_ppo_sft_aflworld_0717_3/_actor/5"
# model_name = "/oss/public/user/liuts/model/Qwen2.5-1.5B-instruct"
# model_name="/cpfs04/user/liutianshuo/math/simpleRL-reason/train/checkpoints/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off12——0525/_actor/3"


# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


# conversations= [
#     #   {"role": "system", "content": "klklkl"},
#       {
#         "role": "assistant",
#         "content": "Thought: I'm at the fridge, which is closed. I'll use it to cool the mug without needing to open it, as the task doesn't specify opening the fridge\nAction: go to shelf 1"
#       },
#     ]

# prompt = tokenizer.apply_chat_template(
#     [{'role': 'system', 'content': "Your are an expert in the ALFRED Embodied Environment."}] +conversations,
#     tokenize=False,
#     add_generation_prompt=True,
#     system_message=""
# )

# print(prompt)
# print(tokenizer.eos_token)


# MODEL_NAME="Qwen2.5_Math_1.5B"
# MODEL="/oss/public/user/liuts/model/${MODEL_NAME}"

# # 加载 tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


# conversations= [
#       {
#         "role": "assistant",
#         "content": "Thought: I'm at the fridge, which is closed. I'll use it to cool the mug without needing to open it, as the task doesn't specify opening the fridge\nAction: go to shelf 1"
#       },
#     ]

# prompt = tokenizer.apply_chat_template(
#     [{'role': 'system', 'content': "Your are an expert in the ALFRED Embodied Environment."}] +conversations,
#     tokenize=False,
#     add_generation_prompt=True,
#     system_message=""
# )

# print(prompt)
# print(tokenizer.eos_token)




# from transformers import AutoTokenizer
# import json

# # 加载 tokenizer（确认路径正确）
# tokenizer = AutoTokenizer.from_pretrained("/oss/public/user/liuts/model/Llama-3.2-3B-Instruct", use_fast=False)

# # 加载 JSON 数据
# with open("/cpfs04/user/liutianshuo/math/simpleRL-reason/train/data/sciworld_train_alfworld_format.json", "r", encoding="utf-8") as f:
#     data = json.load(f)
# with open("/cpfs04/user/liutianshuo/math/simpleRL-reason/train/data/alfworld_sft_mt_2.json", "r", encoding="utf-8") as f:
#     data = json.load(f)


# l=[]
# # 遍历数据并用 tokenizer 构造 prompt
# for sample in data:
#     conversations = [{'role': 'system', 'content': "You are a helpful agent that interacts with the virtual science school environment to solve the given task. "}]+sample["conversations"]  # list of dicts, with 'role' and 'content'
    
#     # 使用 apply_chat_template 方法构造 prompt
#     prompt = tokenizer.apply_chat_template(
#         conversations,
#         tokenize=False,  # 如果想直接输入模型就设为 True
#         add_generation_prompt=True  # 添加 assistant 留白，准备生成
#     )

#     def replace_sys_prompt(text: str) -> str:
#       start_tag = "<|start_header_id|>system<|end_header_id|>"
#       end_tag = "<|eot_id|>"

#       start_index = text.find(start_tag)
#       if start_index == -1:
#           return text  # no system tag found

#       content_start = start_index + len(start_tag)
#       end_index = text.find(end_tag, content_start)
#       if end_index == -1:
#           return text  # no end of system content

#       # Construct the new text
#       new_text = (text[:content_start] +"\nYour are an expert in the ALFRED Embodied Environment.\n" + text[end_index:])
#       return new_text
    
#     prompt=replace_sys_prompt(prompt)

#     encoded = tokenizer(
#       prompt,
#       add_special_tokens=False,  # 不加额外 token（你手动控制的）
#       return_tensors=None
#     )
#     token_count = len(encoded["input_ids"])
#     l.append(token_count)

# print(max(l))

# print("sciworld" in "/cpfs04/user/liutianshuo/math/simpleRL-reason/train/data/sciworld_train_alfworld_format.json")

from scienceworld import ScienceWorldEnv

# 创建环境
env = ScienceWorldEnv()

task_name = "boil"
max_variations = 30

s="Your task is to boil ice to liquid. You should get ice in the kitchen and boil it in the foundry. The objects you can use are metal pot, thermometer, freezer, blast furnace, stove and glass jar. You should pick up a thermometer for temperature measurement. Take actions that will cause it to change its state of matter.  You need to increase the ice's temperature and monitor the temperature closely. Once the ice's state of matter changed, examine the changed state of ice. For compounds without a boiling point, combusting the substance is also acceptable."

for variation_id in range(max_variations):
    try:
        env.load(taskName=task_name, variationIdx=29, generateGoldPath=True)
        env.reset()

   
        # gold_actions = env.getGoalProgressStr()

        # print(f"\n=== Variation {variation_id} ===")
        # if env.get_task_description() ==s:
        print("📝 Task:", env.get_task_description())
        
        # print("📜 Gold Action Sequence:")
        # # for idx, act in enumerate(gold_actions):
        #     # print(f"  {idx+1}. {act}")
        # for idx, action in enumerate(env.get_gold_action_sequence()):
        #     print(f"{idx + 1}. {action}")
        # # print(gold_actions)
    
    except Exception as e:
        print(f"❌ Error on variation {variation_id}: {e}")
        break

env.close()



