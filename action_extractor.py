import re
import ast

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
        # Method 1: pull vector from <answer></answer> blocks
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            vector = self._parse_vector_from_text(answer_content)
            if vector and len(vector) >= 3:
                return vector[:3]
        
        # Method 2: search for explicit key/value pairs
        velocity_dict = {}
        
        # Regex for x_vel_cmd, y_vel_cmd, yaw_vel_cmd
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
        
        # Method 3: fall back to the conclusion section
        return self._extract_from_conclusion(text)
    
    def _parse_vector_from_text(self, text):
        """Parse a vector from text."""
        try:
            # Try parsing literal lists
            if text.startswith('[') and text.endswith(']'):
                return ast.literal_eval(text)
            
            # Otherwise fall back to raw number extraction
            numbers = re.findall(r'[+-]?\d*\.?\d+', text)
            if numbers:
                return [float(num) for num in numbers]
                
        except (ValueError, SyntaxError):
            pass
        
        return None
    
    def _extract_from_conclusion(self, text):
        """Extract velocity hints from concluding sentences."""
        # Search the last conclusion paragraph
        conclusion_patterns = [
            r'(?:conclusion|summary|therefore|thus|final|result).*?$',
            r'(?:robot should|action|command|velocity).*?$'
        ]
        
        for pattern in conclusion_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if matches:
                conclusion = matches[-1]
                
                numbers = re.findall(r'[+-]?\d*\.?\d+', conclusion)
                if len(numbers) >= 3:
                    return [float(numbers[i]) for i in range(3)]
        
        return [0.0, 0.0, 0.0]  # default fallback
    
    def extract_action(self, text):
        """
        Extract discrete action labels from text
        """
        text_lower = text.lower()
        
        # Keyword mapping for each action
        action_keywords = {
            "go forward": ["forward", "ahead", "move forward", "go forward", "advance", "moving forward"],
            "turn right": ["turn right", "rotate right", "clockwise", "turing right", "turning left"],
            "turn left": ["turn left", "rotate left", "counterclockwise"],
            "stop": ["stop", "halt", "pause", "brake", "stand still"],
            "jump": ["jump", "leap", "hop", "bounce", "jumping"],
            "dance": ["dance", "dancing", "groove", "rhythm"],
            "hello": ["hello", "wave", "greet", "greeting"],
            "stretch": ["stretch", "stretching", "extend", "elongate"],
        }
        
        # Score each action by keyword hits
        action_scores = {}
        for action, keywords in action_keywords.items():
            score = 0
            for keyword in keywords:
                score += text_lower.count(keyword)
            if score > 0:
                action_scores[action] = score
        
        # Return the most likely action
        if action_scores:
            return max(action_scores, key=action_scores.get)
        
        # If no keyword matches, infer from velocity vector
        velocity = self.extract_velocity_vector(text)
        return self._infer_action_from_velocity(velocity)
    
    def _infer_action_from_velocity(self, velocity):
        """Infer a discrete action from a velocity vector."""
        if not velocity or len(velocity) < 3:
            return "stop"
        
        x_vel, y_vel, yaw_vel = velocity
        
        # Thresholds for linear and angular motion
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
            return "None"  # default when no action is inferred
    
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

# Example usage
if __name__ == "__main__":
    extractor = ActionExtractor()
    
    # Sample text
    sample_text = """
    Response: lowering of the body is commanded, possibly for stability.
    *   `step_frequency_cmd`: 2.0 Hz (stepping frequency). This is a standard frequency for walking.
    *   `gait`: [0.5, 0.0, 0.0] (trot gait). This indicates a trot gait, which is efficient for forward movement.
    *   `footswing_height_cmd`: 0.173 m (footswing height). The feet are commanded to lift to a moderate height during the swing phase, suitable for general terrain.
    *   `pitch_cmd`, `roll_cmd`: 0.0 (body orientation). No specific pitch or roll adjustments are commanded.
    *   `stance_width_cmd`: 0.189 m (stance width). This is the commanded width between the robot's feet during stance.

    **2. Analyze the Action:**
    *   `body_linear_vel`: [0.319, 0.078, 0.055] m/s. The robot's actual forward velocity (0.319 m/s) is close to the commanded `x_vel_cmd` (0.345 m/s). There's a slight lateral drift (0.078 m/s) and a vertical movement (0.055 m/s), which might be due to gait dynamics or minor terrain variations.

    <answer>[0.34510736, 0.0, 0.39747983, -0.009796846, 2.0, 0.5, 0.0, 0.0, 0.17317529, 0.0, 0.0, 0.18917172]</answer>
    """
    
    result = extractor.extract(sample_text)
    print("Extraction results:")
    print(f"Velocity vector: {result['velocity']}")
    print(f"Action: {result['action']}")