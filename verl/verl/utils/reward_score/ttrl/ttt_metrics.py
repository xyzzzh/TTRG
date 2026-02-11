from collections import Counter
from typing import List
import math
from verl.utils.reward_score.ttrl.auto_extract import auto_extract
from verl.utils.reward_score.ttrl.auto_verify import auto_verify
from collections import defaultdict
import numpy as np
import re

def parse_box(box_str):
    """
    解析模型输出的 bbox 字符串。
    假设格式为 [x1, y1, x2, y2] 或类似格式，归一化到 0-1 或 0-1000 皆可，
    但在计算 variance 时最好统一归一化。
    """
    try:
        # 提取字符串中的所有数字
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", box_str)
        if len(nums) >= 4:
            vals = [float(n) for n in nums[:4]]
            # 简单的归一化处理示例：如果坐标大于1，假设是1000尺度，除以1000
            # 注意：这取决于你的模型输出习惯，请根据实际情况调整
            if any(v > 1.0 for v in vals):
                vals = [v / 1000.0 for v in vals]
            return vals
    except:
        pass
    return None

def compute_iou(box1, box2):
    """计算两个框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 1e-6 else 0.0

def test_time_train_metrics(
    solutions: List[str],
    ground_truth: List[str],
    task="math", extra_info=None):
    
    assert len(solutions) == len(ground_truth), f"{len(solutions)} vs {len(ground_truth)}"
    
    # --- Spatial-TTRV 逻辑开始 ---
    if task == "grounding":
        # 1. 解析所有预测框
        parsed_boxes = [parse_box(s) for s in solutions]
        # 记录有效框的索引
        valid_indices = [i for i, b in enumerate(parsed_boxes) if b is not None]
        valid_boxes = [parsed_boxes[i] for i in valid_indices]
        
        # 初始化奖励为 0 (针对解析失败的情况)
        rewards_en = [0.0] * len(solutions)
        
        consensus_score_mean = 0.0
        uncertainty_penalty = 0.0

        if len(valid_boxes) > 1:
            # 2. 计算空间共识 (Frequency Analog -> Spatial Consensus)
            n_valid = len(valid_boxes)
            iou_matrix = np.zeros((n_valid, n_valid))
            for i in range(n_valid):
                for j in range(n_valid):
                    if i == j:
                        iou_matrix[i, j] = 1.0
                    else:
                        iou_matrix[i, j] = compute_iou(valid_boxes[i], valid_boxes[j])
            
            # 每个框的得分是它与其他所有框的平均 IoU
            # 越高说明这个框处于“共识中心”
            spatial_consensus_scores = np.mean(iou_matrix, axis=1) 
            consensus_score_mean = np.mean(spatial_consensus_scores)

            # 3. 计算几何不确定性 (Entropy Analog -> Geometric Variance)
            # 计算坐标的方差作为不确定性惩罚
            box_array = np.array(valid_boxes) # shape (N, 4)
            # 计算坐标的标准差或方差均值
            coord_std = np.std(box_array, axis=0).mean()
            uncertainty_penalty = coord_std 

            # 4. 组合最终奖励
            # r = r_SC - alpha * r_GU
            # alpha 系数需要根据方差的量级调节，这里设为 1.0 作为示例
            alpha = 1.0 
            
            for local_idx, global_idx in enumerate(valid_indices):
                r_sc = spatial_consensus_scores[local_idx]
                # 最终奖励：共识越高越好，方差越低越好
                rewards_en[global_idx] = r_sc - (alpha * uncertainty_penalty)
        
        # --- 计算评估指标 (Metrics) ---
        # 解析 Ground Truth (假设 GT 只有一个且格式正确)
        gt_box = parse_box(ground_truth[0]) if isinstance(ground_truth[0], str) else ground_truth[0]
        
        pred_ious = []
        for b in parsed_boxes:
            if b is not None and gt_box is not None:
                pred_ious.append(compute_iou(b, gt_box))
            else:
                pred_ious.append(0.0)
        
        # 计算 Average IoU 和 Accuracy@0.5
        avg_iou = sum(pred_ious) / len(pred_ious) if pred_ious else 0.0
        acc_05 = sum(1.0 for iou in pred_ious if iou >= 0.5) / len(pred_ious) if pred_ious else 0.0

        ttrl_metrics = {
            "avg_iou": avg_iou,
            "acc_05": acc_05,
            "spatial_consensus": consensus_score_mean,
            "coord_uncertainty": uncertainty_penalty,
            "valid_box_ratio": len(valid_boxes) / len(solutions)
        }
        
        return rewards_en, ttrl_metrics
    # --- Spatial-TTRV 逻辑结束 ---

    assert len(set(ground_truth)) == 1, f"Ground truth is not unique: {ground_truth}"
    ground_truth = ground_truth[0]

    model_answers = auto_extract(task, solutions, extra_info=extra_info)
    counter = Counter(model_answers)
    total = len(model_answers)
    reward_p = [counter[ans] / total for ans in model_answers]


    entropy = 0.0
    for count in counter.values():
        probability = count / total
        if probability > 0:  # Avoid log(0)
            entropy -= probability * math.log(probability)
    
    if total > 1:
        max_entropy = math.log(len(counter))  # Max entropy for this many unique answers
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    else:
        normalized_entropy = 0.0
    
    estimated_label, majority_count = counter.most_common(1)[0]
    
    
    hit_rate = 1.0 if auto_verify(task, [estimated_label], [ground_truth], extra_info=extra_info)[0][0] else 0.0
    majority_ratio = majority_count / len(solutions)
    

    rewards, _ = auto_verify(task, solutions, [estimated_label] * len(solutions), extra_info=extra_info)
    true_rewards, _ = auto_verify(task, solutions, [ground_truth] * len(solutions), extra_info=extra_info)
    rewards_en = [(r*1) - (0.75 * normalized_entropy) for r in reward_p]
    
    rewards_hit_rate = 0
    for reward, true_reward in zip(rewards, true_rewards):
        if reward == true_reward:
            rewards_hit_rate += 1
    rewards_hit_rate = rewards_hit_rate / len(rewards)

    assert len(rewards) == len(solutions), f"{len(rewards)} vs {len(solutions)}"

    ttrl_metrics = {
        "label_accuracy": hit_rate,
        "reward_accuracy": rewards_hit_rate,
        "majority_ratio": majority_ratio,
        "ground_truth_ratio": sum(true_rewards) / len(true_rewards),
        "majority_voting_reward": sum(rewards) / len(rewards),
        f"pass@{len(solutions)}": 1.0 if sum(true_rewards) >= 1 else 0.0,
    }
    return rewards_en, ttrl_metrics

def post_test_time_train_metrics(
    solutions: List[str],
    ground_truth: List[str],
    pred_rewards: List,
    task="math", extra_info=None):
    assert len(solutions) == len(ground_truth), f"{len(solutions)} vs {len(ground_truth)}"
    assert len(solutions) == len(pred_rewards), f"{len(solutions)} vs {len(pred_rewards)}"
    assert len(set(ground_truth)) == 1, f"Ground truth is not unique: {ground_truth}"
    ground_truth = ground_truth[0]

    model_answers = auto_extract(task, solutions, extra_info=extra_info)

    # counter = Counter(model_answers)
    
    # true_label_ratio = counter.get(ground_truth, 0) / len(solutions)

    true_rewards, _ = auto_verify(task, solutions, [ground_truth] * len(solutions), extra_info=extra_info)

    # Compare pred_rewards with true_rewards to calculate reward hit rate
    rewards_hit_rate = sum(
        1 if pred == true else 0 for pred, true in zip(pred_rewards, true_rewards)
    ) / len(pred_rewards)



    post_ttrl_metrics = {
        "post_reward_accuracy": rewards_hit_rate,
        "post_ground_truth_ratio": sum(true_rewards) / len(true_rewards),
        f"post_pass@{len(solutions)}": 1.0 if sum(true_rewards) > 0 else 0.0,
    }
    return post_ttrl_metrics