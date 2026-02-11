from collections import Counter
from typing import List
import math
from verl.utils.reward_score.ttrl.auto_extract import auto_extract
from verl.utils.reward_score.ttrl.auto_verify import auto_verify
from collections import defaultdict
import numpy as np
import re

# --- 配置项 ---
IMG_W = 640.0
IMG_H = 480.0
# -------------

def process_coords(vals):
    """
    统一的坐标后处理逻辑：归一化 + 裁剪 + 几何约束
    """
    if not vals or len(vals) < 4:
        return None
    
    vals = [float(v) for v in vals[:4]]
    
    # --- 归一化逻辑 (Resolution-Agnostic) ---
    # 如果坐标数值看起来像绝对坐标 (有一个大于1)，则执行除法
    # 这样兼容了模型偶尔输出 0-1 小数的情况
    if any(v > 1.0 for v in vals):
        vals[0] /= IMG_W
        vals[1] /= IMG_H
        vals[2] /= IMG_W
        vals[3] /= IMG_H
    
    # 裁剪到 0-1
    vals = [min(max(v, 0.0), 1.0) for v in vals]
    
    # 几何约束：确保 x1<x2, y1<y2
    if vals[0] > vals[2]: vals[0], vals[2] = vals[2], vals[0]
    if vals[1] > vals[3]: vals[1], vals[3] = vals[3], vals[1]
    
    return vals

def parse_box(box_str):
    """
    鲁棒的解析函数，支持 JSON, Markdown, 和纯文本列表
    """
    box_str = box_str.strip()
    
    # === 策略 1: 尝试作为 JSON 解析 (最准确) ===
    # 1.1 去除 Markdown 标记 ```json ... ```
    clean_json_str = box_str
    if "```" in clean_json_str:
        # 提取 ```json 和 ``` 之间的内容，或者 ``` 和 ``` 之间的内容
        pattern = r"```(?:json)?(.*?)```"
        match = re.search(pattern, clean_json_str, re.DOTALL)
        if match:
            clean_json_str = match.group(1).strip()
    
    try:
        data = json.loads(clean_json_str)
        
        # 处理 list 包裹 [{"bbox_2d":...}] 的情况
        if isinstance(data, list):
            if len(data) > 0:
                data = data[0] # 取第一个预测对象
            else:
                return None
                
        # 提取坐标，支持多种常见的 key
        if isinstance(data, dict):
            for key in ["bbox_2d", "bbox", "box", "coordinates"]:
                if key in data:
                    return process_coords(data[key])
        
        # 如果就是一个列表 [x, y, x, y]
        if isinstance(data, list) and len(data) == 4:
            return process_coords(data)
            
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass # JSON 解析失败，进入策略 2

    # === 策略 2: 基于 List 结构的 Regex (次选) ===
    # 专门匹配 [num, num, num, num] 这种结构，避免匹配到 "bbox_2d" 里的数字
    # 允许空格、换行
    list_pattern = r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]"
    match = re.search(list_pattern, box_str)
    if match:
        return process_coords([float(x) for x in match.groups()])

    # === 策略 3: 暴力提取 (兜底，风险最高) ===
    # 只有当前面都失败时才用，且尝试跳过可能的 "bbox_2d" 干扰
    # 这里的逻辑稍微保守一点：只提取冒号 : 后面的数字，或者是列表里的数字
    try:
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", box_str)
        if len(nums) >= 4:
            # 简单的启发式过滤：如果第一个数字特别小(比如2)且后面跟了4个大数，可能那个2是干扰项
            # 但最安全的是相信 JSON 解析。这里只做最后尝试。
            # 针对 "bbox_2d" 这种常见干扰，如果我们找到了5个数字，且第一个是2，我们可以尝试丢弃第一个
            if len(nums) == 5 and nums[0] == '2' and "2d" in box_str:
                 return process_coords(nums[1:])
            
            return process_coords(nums[:4])
    except Exception:
        pass

    return None

def compute_iou(box1, box2):
    """计算两个框的 IoU (输入已归一化到 0-1)"""
    if box1 is None or box2 is None:
        return 0.0
        
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 面积 = 宽 * 高
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    # 防止除以零
    return intersection / union if union > 1e-6 else 0.0

def test_time_train_metrics(
    solutions: List[str],
    ground_truth: List[str],
    task="math", extra_info=None):
    
    assert len(solutions) == len(ground_truth), f"{len(solutions)} vs {len(ground_truth)}"
    
    # --- Spatial-TTRV 逻辑开始 ---
    if task == "grounding":
        # 1. 解析所有预测框 (此时已转为 0-1 相对坐标)
        parsed_boxes = [parse_box(s) for s in solutions]
        valid_indices = [i for i, b in enumerate(parsed_boxes) if b is not None]
        valid_boxes = [parsed_boxes[i] for i in valid_indices]
        
        rewards_en = [0.0] * len(solutions)
        consensus_score_mean = 0.0
        uncertainty_penalty = 0.0
        
        if len(valid_boxes) > 1:
            # 2. 计算空间共识 (Spatial Consensus)
            n_valid = len(valid_boxes)
            iou_matrix = np.zeros((n_valid, n_valid))
            for i in range(n_valid):
                for j in range(n_valid):
                    if i == j:
                        iou_matrix[i, j] = 1.0
                    else:
                        iou_matrix[i, j] = compute_iou(valid_boxes[i], valid_boxes[j])
            
            spatial_consensus_scores = np.mean(iou_matrix, axis=1)
            consensus_score_mean = np.mean(spatial_consensus_scores)

            # 3. 计算几何不确定性 (Geometric Uncertainty)
            box_array = np.array(valid_boxes)
            # 计算坐标的标准差 (0-1尺度下，通常在 0.01~0.2 之间)
            coord_std = np.std(box_array, axis=0).mean()
            uncertainty_penalty = coord_std
            
            # 4. 组合奖励
            # 这里的 alpha=2.0 在 0-1 空间下是比较合理的经验值
            # 如果不归一化，这里的 alpha 可能需要设为 0.005
            alpha = 2.0  
            
            for local_idx, global_idx in enumerate(valid_indices):
                r_sc = spatial_consensus_scores[local_idx]
                rewards_en[global_idx] = r_sc - (alpha * uncertainty_penalty)

            # 5. 计算评估指标 (与 GT 对比)
            # 假设 Ground Truth 也是绝对坐标字符串，同样通过 parse_box 归一化
            gt_box = parse_box(ground_truth[0]) if isinstance(ground_truth[0], str) else ground_truth[0]
            
            pred_ious = []
            for b in parsed_boxes:
                if b is not None and gt_box is not None:
                    pred_ious.append(compute_iou(b, gt_box))
                else:
                    pred_ious.append(0.0)
            
            avg_iou = sum(pred_ious) / len(pred_ious) if pred_ious else 0.0
            acc_05 = sum(1.0 for iou in pred_ious if iou >= 0.5) / len(pred_ious) if pred_ious else 0.0
            
            # 将归一化后的 Std 还原为像素 Std 用于日志显示 (可选)
            pixel_std_display = uncertainty_penalty * ((IMG_W + IMG_H) / 2)

            ttrl_metrics = {
                "avg_iou": avg_iou,
                "acc_05": acc_05,
                "spatial_consensus": consensus_score_mean,
                "coord_uncertainty_norm": uncertainty_penalty,
                "pixel_std_approx": pixel_std_display, # 方便您直观理解像素级抖动
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
    
    # --- Spatial-TTRV Logic (New) ---
    if task == "grounding":
        # 1. 解析 Ground Truth (假设一个 Batch 内是对同一个 Prompt 的多次采样，GT 应该是唯一的)
        # 注意：使用与训练时相同的 parse_box 以确保归一化逻辑一致
        gt_box = parse_box(ground_truth[0]) if isinstance(ground_truth[0], str) else ground_truth[0]
        
        # 2. 解析预测结果
        parsed_boxes = [parse_box(s) for s in solutions]
        
        # 3. 计算与 GT 的 IoU
        pred_ious = []
        for b in parsed_boxes:
            if b is not None and gt_box is not None:
                pred_ious.append(compute_iou(b, gt_box))
            else:
                pred_ious.append(0.0)
        
        # 4. 计算统计指标
        avg_iou = sum(pred_ious) / len(pred_ious) if pred_ious else 0.0
        acc_05 = sum(1.0 for iou in pred_ious if iou >= 0.5) / len(pred_ious) if pred_ious else 0.0
        pass_at_k = 1.0 if any(iou >= 0.5 for iou in pred_ious) else 0.0
        
        # 5. 奖励一致性 (可选分析：模型预测的高分是否对应高 IoU)
        # 这里简单计算一下预测奖励与真实 IoU 的相关性，或者只是记录平均值
        
        return {
            "post_avg_iou": avg_iou,
            "post_acc_05": acc_05,
            "post_pass@k": pass_at_k,
            # 可以添加更多针对 Grounding 的指标，如 Acc@0.75 等
        }
    # --------------------------------
    
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