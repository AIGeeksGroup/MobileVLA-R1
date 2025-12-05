from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os, re, random, time, requests, ast
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset
from ref_server import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list
from llava.data.nav_cot_utils import normalize_nav_path
from pathlib import Path

# Configuration
model_path = "/root/autodl-tmp/model/finetune/llm"
# gen_device = 1    # GPU device for generation
Q_batch_size = 2
num_pre_Q = 3
train_batch_size = 2
compute_gen_logps = True
ref_server = "http://localhost:59875"
dataset_source = os.environ.get("GEN_DATASET", "/root/autodl-tmp/dataset/CoT")
navcot_image_root = os.environ.get("NAVCOT_IMAGE_ROOT")
navcot_depth_root = os.environ.get("NAVCOT_DEPTH_ROOT")
navcot_use_depth = bool(navcot_depth_root)
navcot_use_point = os.environ.get("NAVCOT_USE_POINT", "1") == "1"
navcot_depth_frames = int(os.environ.get("NAVCOT_DEPTH_FRAMES", 1))
navcot_pointcloud_points = int(os.environ.get("NAVCOT_POINT_POINTS", 2048))

# Directory used to exchange model checkpoints with the trainer
model_update_dir = "/root/autodl-tmp/model_updates"
os.makedirs(model_update_dir, exist_ok=True)

class ActionExtractor:
    def __init__(self):
        self.action_space = [
            "go forward", "turn right", "turn left", "stop", "jump", 
            "dance", "hello", "stretch"
        ]
    
    def extract_velocity_vector(self, text):
        """
        Extract the first three entries of the velocity vector [x_vel_cmd, y_vel_cmd, yaw_vel_cmd]
        """
        # Method 1: read values from <answer></answer> blocks
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            vector = self._parse_vector_from_text(answer_content)
            if vector and len(vector) >= 3:
                return vector[:3]
        
        # Method 2: search explicit key/value pairs
        velocity_dict = {}
        
        # Regex for each velocity field
        patterns = {
            'x_vel_cmd': r'x_vel_cmd[:\s]*([+-]?\d*\.?\d+)',
            'y_vel_cmd': r'y_vel_cmd[:\s]*([+-]?\d*\.?\d+)', 
            'yaw_vel_cmd': r'yaw_vel_cmd[:\s]*([+-]?\d*\.?\d+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                velocity_dict[key] = float(match.group(1))
        
        if len(velocity_dict) == 3:
            return [
                velocity_dict.get('x_vel_cmd', 0.0),
                velocity_dict.get('y_vel_cmd', 0.0), 
                velocity_dict.get('yaw_vel_cmd', 0.0)
            ]
        
        # Method 3: fallback to concluding sentences
        return self._extract_from_conclusion(text)
    
    def _parse_vector_from_text(self, text):
        """Parse a vector from raw text."""
        try:
            # Attempt literal list parsing
            if text.startswith('[') and text.endswith(']'):
                return ast.literal_eval(text)
            
            # Otherwise collect every number
            numbers = re.findall(r'[+-]?\d*\.?\d+', text)
            if numbers:
                return [float(num) for num in numbers]
                
        except (ValueError, SyntaxError):
            pass
        
        return None
    
    def _extract_from_conclusion(self, text):
        """Extract velocity hints from concluding sentences."""
        # Grab the last matching conclusion paragraph
        conclusion_patterns = [
            r'(?:conclusion|summary|therefore|thus|final|result).*?$',
            r'(?:robot should|action|command|velocity).*?$'
        ]
        
        for pattern in conclusion_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if matches:
                conclusion = matches[-1]
                
                # Extract numbers from that paragraph
                numbers = re.findall(r'[+-]?\d*\.?\d+', conclusion)
                if len(numbers) >= 3:
                    return [float(numbers[i]) for i in range(3)]
        
        return [0.0, 0.0, 0.0]  # default fallback
    
    def extract_action(self, text):
        """
        Extract discrete action labels from text
        """
        text_lower = text.lower()
        
        # Keywords per action
        action_keywords = {
            "go forward": ["forward", "ahead", "move forward", "go forward", "advance", "moving forward"],
            "turn right": ["turn right", "rotate right", "clockwise", "turning right"],
            "turn left": ["turn left", "rotate left", "counterclockwise", "turning left"],
            "stop": ["stop", "halt", "pause", "brake", "stand still"],
            "jump": ["jump", "leap", "hop", "bounce", "jumping"],
            "dance": ["dance", "dancing", "groove", "rhythm"],
            "hello": ["hello", "wave", "greet", "greeting"],
            "stretch": ["stretch", "stretching", "extend", "elongate"],
        }
        
        # Count keyword hits
        action_scores = {}
        for action, keywords in action_keywords.items():
            score = 0
            for keyword in keywords:
                score += text_lower.count(keyword)
            if score > 0:
                action_scores[action] = score
        
        # Pick the highest scoring action
        if action_scores:
            return max(action_scores, key=action_scores.get)
        
        # Otherwise infer from velocity vector
        velocity = self.extract_velocity_vector(text)
        return self._infer_action_from_velocity(velocity)
    
    def _infer_action_from_velocity(self, velocity):
        """Infer discrete action from the velocity vector."""
        if not velocity or len(velocity) < 3:
            return "stop"
        
        x_vel, y_vel, yaw_vel = velocity
        
        # Thresholds for motion
        linear_threshold = 0.1
        angular_threshold = 0.2
        
        if abs(x_vel) < linear_threshold and abs(y_vel) < linear_threshold and abs(yaw_vel) < angular_threshold:
            return "stop"
        elif x_vel > linear_threshold:
            return "go forward"
        elif yaw_vel > angular_threshold:
            return "turn left"
        elif yaw_vel < -angular_threshold:
            return "turn right"
        else:
            return None  # no confident action
    
    def extract(self, text):
        """
        Main entry: return both velocity vector and action label
        """
        velocity = self.extract_velocity_vector(text)
        action = self.extract_action(text)
        
        return {
            "velocity": velocity,
            "action": action
        }

class RewardCalculator:
    """Reward calculator for GRPO training."""
    
    def __init__(self, action_space=None):
        if action_space is None:
            action_space = [
                "go forward", "turn right", "turn left", "stop", "jump", 
                "dance", "hello", "stretch"
            ]
        self.action_space = action_space
    
    def format_reward(self, response_text):
        """
        Format reward: ensure the response contains <answer>[12 floats]</answer>.
        Returns 1 if formatted correctly, otherwise 0.
        """
        try:
            # 1. Check <answer></answer>
            answer_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
            if not answer_match:
                return 0.0
            
            answer_content = answer_match.group(1).strip()
            
            # 2. Ensure it looks like a list
            if not (answer_content.startswith('[') and answer_content.endswith(']')):
                return 0.0
            
            # 3. Parse into numeric list
            try:
                vector = ast.literal_eval(answer_content)
                
                if not isinstance(vector, list):
                    return 0.0
                
                if len(vector) != 12:
                    return 0.0
                
                for item in vector:
                    if not isinstance(item, (int, float)):
                        return 0.0
                
                return 1.0
                
            except (ValueError, SyntaxError):
                return self._manual_parse_check(answer_content)
                
        except Exception as e:
            print(f"Format check failed: {e}")
            return 0.0
    
    def _manual_parse_check(self, answer_content):
        """
        Fallback parser if literal_eval fails.
        """
        try:
            content = answer_content[1:-1].strip()
            
            items = [item.strip() for item in content.split(',')]
            
            if len(items) != 12:
                return 0.0
            
            for item in items:
                try:
                    float(item)
                except ValueError:
                    return 0.0
            
            return 1.0
            
        except Exception:
            return 0.0
    
    def multiply_reward(self, extracted_result, ground_truth_vector):
        """
        Dot-product reward between predicted and GT velocity vectors.
        """
        velocity = extracted_result.get('velocity')
        if velocity is None or len(velocity) != 3:
            return 0.0
        
        try:
            pred_vector = np.array(velocity, dtype=np.float32)
            gt_vector = np.array(ground_truth_vector[:3], dtype=np.float32)
            
            dot_product = np.dot(pred_vector, gt_vector)
            
            pred_norm = np.linalg.norm(pred_vector)
            gt_norm = np.linalg.norm(gt_vector)
            
            if pred_norm > 1e-8 and gt_norm > 1e-8:
                cosine_sim = dot_product / (pred_norm * gt_norm)
                return (cosine_sim + 1) / 2
            else:
                if pred_norm < 1e-8 and gt_norm < 1e-8:
                    return 1.0  # both zero vectors
                else:
                    return 0.0  # mismatch
                
        except Exception as e:
            print(f"Failed to compute dot-product reward: {e}")
            return 0.0
    
    def action_reward(self, extracted_result):
        """
        Action reward: 1 if action in action_space, else 0.
        """
        action = extracted_result.get('action')
        if action is not None and action in self.action_space:
            return 1.0
        return 0.0
    
    def combined_reward(self, response_text, extracted_result, ground_truth_vector, weights=[1.0, 1.0, 1.0]):
        """
        Combined reward with optional weightings; returns total and components.
        """
        format_r = self.format_reward(response_text)
        multiply_r = self.multiply_reward(extracted_result, ground_truth_vector)
        action_r = self.action_reward(extracted_result)
        
        total_reward = (weights[0] * format_r + 
                       weights[1] * multiply_r + 
                       weights[2] * action_r)
        
        return total_reward, {
            'format': format_r,
            'multiply': multiply_r, 
            'action': action_r,
            'total': total_reward
        }

def load_custom_dataset(dataset_dir, split="train"):
    """
    dataset_dir structure:
        datasets/
        ├── train/                # image folder
        ├── annotations.json      # metadata
    split:
        currently only 'train' is used but can be extended.
    """
    if os.path.isfile(dataset_dir) and dataset_dir.endswith(".jsonl"):
        records = []
        with open(dataset_dir, "r", encoding="utf-8") as f:
            for line in f:
                ann = json.loads(line)
                records.append(
                    {
                        "video_id": ann.get("scene_id"),
                        "q": ann.get("instruction"),
                        "a": f"<think>{ann.get('think', '')}</think>\n<action>{ann.get('action', '')}</action>",
                        "frames": ann.get("images", []),
                        "navcot": ann,
                    }
                )
        return Dataset.from_list(records)

    ann_path = os.path.join(dataset_dir, "annotations.json")
    with open(ann_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    records = []
    for ann in annotations:
        frames = ann.get("frames", [])
        images = []
        for frame_path in frames:
            img_path = os.path.join(dataset_dir, split, frame_path)
            if os.path.exists(img_path):
                images.append(img_path)
            else:
                print(f"[WARN] Image not found: {img_path}")

        records.append({
            "video_id": ann["video_id"],
            "q": ann["q"],
            "a": ann["a"],
            "frames": images,
        })

    dataset = Dataset.from_list(records)
    return dataset


def resolve_navcot_frames(sample):
    frames = sample.get("frames", [])
    if navcot_image_root:
        return [normalize_nav_path(frame, navcot_image_root) for frame in frames]
    return frames


def resolve_navcot_depth_inputs(sample):
    if not navcot_use_depth or not navcot_depth_root:
        return None
    nav_meta = sample.get("navcot")
    if not nav_meta:
        return None
    frames = nav_meta.get("images", [])
    if not frames:
        return None
    depth_candidates = frames[-navcot_depth_frames:]
    depth_paths = []
    for frame in depth_candidates:
        depth_paths.append(normalize_nav_path(frame, navcot_depth_root))
    return depth_paths

def parse_ground_truth_vector(answer_text):
    """
    Parse a 12-D ground-truth vector from text.
    """
    try:
        if answer_text.startswith('[') and answer_text.endswith(']'):
            vector = ast.literal_eval(answer_text)
            if isinstance(vector, list) and len(vector) == 12:
                return vector
        
        numbers = re.findall(r'[+-]?\d*\.?\d+', answer_text)
        if len(numbers) >= 12:
            return [float(num) for num in numbers[:12]]
        
        return [0.0] * 12
        
    except Exception as e:
        print(f"Failed to parse ground-truth vector: {e}")
        return [0.0] * 12

def main():
    # Initialize tokenizer/model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda()

    ref_server_ver = 'tensor'
    num_return_sequences = num_pre_Q

    dataset = load_custom_dataset(dataset_source)
    QAs = [{'Q': item['q'], 'A': item['a'], 'images': item['frames']} for item in dataset]

    system_prompt = """You are a Unitree Go2 robot dog, given an instruction and observation, output exactly one list [x_vel_cmd, y_vel_cmd, yaw_vel_cmd, body_height_cmd, step_frequency_cmd, gait1, gait2, gait3, footswing_height_cmd, pitch_cmd, roll_cmd, stance_width_cmd] where all values are floats. Please provide your reasoning in <think></think> tags and your final answer in <answer></answer> tags with exactly 12 numerical values."""

    from inference import NaVILAImageInference
    inferencer = NaVILAImageInference(
        model_path="/root/autodl-tmp/model/finetune",
        lora_path=None,
        device="cuda",
        pointcloud_points=navcot_pointcloud_points,
    )

    # Initialize extractors and reward calculator
    extractor = ActionExtractor()
    reward_calculator = RewardCalculator()

    def gen_answers(inputs):
        """
        inputs: List[dict] each containing Q and image metadata
        """
        answers = []
        ans_token_ids = []

        for item in inputs:
            q = item["Q"]
            images = resolve_navcot_frames(item)
            if not images:
                images = item["images"]
            depth_inputs = resolve_navcot_depth_inputs(item)
            point_inputs = "from_depth" if (navcot_use_point and depth_inputs) else None
            for _ in range(num_return_sequences):
                output_text, output_ids = inferencer.generate_response(
                    image_input=images,
                    question=f"{system_prompt}\n{q}",
                    max_new_tokens=4096,
                    temperature=0.9,
                    top_p=0.9,
                    do_sample=True,
                    depth_input=depth_inputs,
                    point_cloud=point_inputs,
                    return_token_ids=True,
                )
                answers.append(output_text)
                ans_token_ids.append(output_ids)
            
            print(f"answers = {answers}")

        return answers, ans_token_ids

    def gen_samples(inputs):
        prompts = [x["Q"] for x in inputs]
        answers, ans_token_ids = gen_answers(inputs)
        rewards = []
        
        for i, inp in enumerate(inputs):
            # Parse ground truth vector
            ground_truth_vector = parse_ground_truth_vector(inp["A"])
            
            for a in answers[i*num_pre_Q:(i+1)*num_pre_Q]:
                # Extract velocity/action info
                extracted = extractor.extract(a)
                
                # Compute combined reward
                total_reward, reward_details = reward_calculator.combined_reward(
                    a, extracted, ground_truth_vector, 
                    weights=[1.0, 2.0, 1.0]  # Adjust component weights as needed
                )
                
                rewards.append(total_reward)
                
                # Optionally print reward breakdowns for debugging
                if i == 0 and len(rewards) <= 2:
                    print(f"Reward details: {reward_details}")
        
        prompts_text = [tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True) for x in prompts]
        return prompts_text, torch.tensor(rewards, dtype=torch.bfloat16), answers, ans_token_ids

    def try_update_model():
        update_file = os.path.join(model_update_dir, "latest_model.pt")
        flag_file = os.path.join(model_update_dir, "update_ready.flag")
        
        if os.path.exists(flag_file):
            try:
                print('[GEN WORKER] Loading new model ...')
                state_dict = torch.load(update_file, map_location='cuda')
                model.load_state_dict(state_dict, strict=False)
                print('[GEN WORKER] Model updated')
                os.remove(flag_file)
                del state_dict
                torch.cuda.empty_cache()
            except Exception as e:
                print(f'[GEN WORKER] Failed to update model: {e}')

    # Main loop
    for it in range(999999999):
        print("=== it", it)
        if it % 3 == 0: try_update_model()
        
        try:
            inputs = random.sample(QAs, Q_batch_size)
            tic = time.time()
            print("\nGenerating answers...\n")
            prompt_inputs, rewards, answers, ans_token_ids = gen_samples(inputs)
        except Exception as e:
            print("Error during generation:", e)
            continue
            
        print(f'time: {time.time()-tic:.2f}s    ', 'rewards:', rewards)
        if it % 5 == 0: print('answers:', answers[0])

        for i, pp in enumerate(prompt_inputs):
            prompt_ids = tokenizer(pp, return_tensors="pt", add_special_tokens=False)["input_ids"]
            plen = prompt_ids.shape[1]
            curr_answers = answers[i*num_pre_Q:(i+1)*num_pre_Q]
            curr_ans_ids = ans_token_ids[i*num_pre_Q:(i+1)*num_pre_Q]
            curr_rewards = rewards[i*num_pre_Q:(i+1)*num_pre_Q]
            if curr_rewards.max() - curr_rewards.min() < 1e-4: continue

            if ref_server_ver == 'tensor':
                curr_rewards = (curr_rewards - curr_rewards.mean()) / (curr_rewards.std() + 1e-4)
                for ii in range(0, num_pre_Q, train_batch_size):
                    sub_rewards = curr_rewards[ii:ii+train_batch_size]
                    sub_ans_ids = curr_ans_ids[ii:ii+train_batch_size]
                    tensor_list = [torch.tensor(lst) for lst in sub_ans_ids]
                    output_ids = pad_sequence(tensor_list, batch_first=True, padding_value=tokenizer.pad_token_id)
                    Qrep = prompt_ids.repeat(1, output_ids.shape[0]).view(-1, plen)
                    merged_ids = torch.cat([Qrep, output_ids], dim=1)
                    data = [json.dumps({"plen": plen}).encode(), tensor_to_bytes(merged_ids), tensor_to_bytes(sub_rewards)]

                    if compute_gen_logps:
                        with torch.no_grad():
                            logits = model(merged_ids.to(model.device)).logits
                            log_probs = torch.log_softmax(logits, dim=-1)
                            seq_logps = []
                            for b in range(log_probs.shape[0]):
                                tok_ids = merged_ids[b, plen:]
                                seq_logps.append([log_probs[b, plen+i-1, tid].item() for i, tid in enumerate(tok_ids, start=1)])
                            gen_logps = torch.tensor(seq_logps)
                        data.append(tensor_to_bytes(gen_logps))

                    xdata = make_bytes_list(data)
                    r = requests.post(f"{ref_server}/upload", data=xdata)
                    if r.content == b'string': ref_server_ver = 'string'
            elif ref_server_ver == 'string':
                xdata = make_bytes_list([json.dumps({"Q": pp[0], "As": curr_answers}).encode(),
                                        tensor_to_bytes(curr_rewards)])
                r = requests.post(f"{ref_server}/upload", data=xdata)
                if r.content == b'tensor': ref_server_ver = 'tensor'

if __name__ == '__main__':
    main()