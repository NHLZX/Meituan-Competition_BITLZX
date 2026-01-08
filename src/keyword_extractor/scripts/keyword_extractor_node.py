import os
import json
import sys
import re

# --- 环境清理 ---
proxies_to_clear = ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy']
for key in proxies_to_clear:
    if key in os.environ:
        os.environ.pop(key)

try:
    import ollama
except ImportError:
    print("错误: 未找到 ollama 库。请先运行: pip install ollama")
    sys.exit(1)

class NavigationLLM:
    def __init__(self, model_name="qwen2.5:14b"):
        self.model = model_name
        print(f"[初始化] 正在加载强力模型: {self.model}")
        print("[模式] 纯语义理解 + 智能JSON拆包")

    def extract_list_from_json(self, data):
        """
        核心修复：自动从各种 JSON 结构中提取列表
        兼容:
        1. ["front", "tree"]
        2. {"actions": ["front", "tree"]}
        3. {"result": [...]}
        """
        if isinstance(data, list):
            return data
        
        if isinstance(data, dict):
            # 优先查找可能的键名
            for key in ['actions', 'sequence', 'steps', 'commands', 'output']:
                if key in data and isinstance(data[key], list):
                    return data[key]
            
            # 如果没找到常见键名，遍历所有值，找第一个是 list 的
            for val in data.values():
                if isinstance(val, list):
                    return val
        
        return []

    def parse_instruction(self, instruction):
        system_prompt = """
        You are an advanced robot navigation parsing engine. 
        Your goal is to translate natural language instructions into a strict sequence of [Action, Marker] pairs.

        ### 1. The World
        - **Allowed Actions (4)**: "front", "back", "left", "right"
        - **Allowed Markers (8)**: "tree", "traffic cone", "bench", "billboard", "trash", "barrel", "fire hydrant", "tractor trailer"
        - **IGNORED Entities**: "pedestrian", "human", "person", "pass", "avoid", "bypass"

        ### 2. Semantic Logic Rules (Apply these strictly)
        1. **Strict Pairs**: Output MUST be a flat list of pairs: `[Action, Marker, Action, Marker...]`.
        2. **Turn Overrides Move**: 
           - If instruction says "Turn right and go to the tree", the effective action is "right". 
           - RESULT: `["right", "tree"]` (NOT `["right", "front", "tree"]`).
           - "front" is ONLY used if there is NO turn command before the target marker.
        3. **Ignore Obstacles**: 
           - "Bypass pedestrians and go to the cone" -> The only relevant part is "go to cone". 
           - RESULT: `["front", "traffic cone"]` (Pedestrians are invisible to the state machine).
        4. **Implicit Action**: 
           - "Stop at the bench" -> Implicitly means move front to it. 
           - RESULT: `["front", "bench"]`.

        ### 3. Output Format
        - Return ONLY the JSON list. Do not wrap it in an object.
        - Example: ["front", "tree", "right", "bench"]
        
        ### 4. Few-Shot Logic Examples (Reasoning -> Output)

        User: "Move forward and stop at the tree."
        Logic: No turn mentioned. Default action is "front". Target is "tree".
        Output: ["front", "tree"]

        User: "Move forward to the tree, turn right, go straight and stop at the traffic cone."
        Logic: 
        1. To "tree": No turn -> "front". Pair: ["front", "tree"]
        2. To "traffic cone": "turn right" is the dominant action. It overrides "go straight". Pair: ["right", "traffic cone"]
        Output: ["front", "tree", "right", "traffic cone"]

        User: "Walk forward to the billboard, then turn right, bypass pedestrians and continue to the traffic cone."
        Logic:
        1. To "billboard": No turn -> "front". Pair: ["front", "billboard"]
        2. To "traffic cone": "turn right". "Bypass pedestrians" is noise/obstacle avoidance (IGNORE). Pair: ["right", "traffic cone"]
        Output: ["front", "billboard", "right", "traffic cone"]

        User: "Turn back, move forward to the tree, turn right, walk until you see a traffic cone."
        Logic:
        1. To "tree": "turn back". Overrides "move forward". Pair: ["back", "tree"]
        2. To "traffic cone": "turn right". Pair: ["right", "traffic cone"]
        Output: ["back", "tree", "right", "traffic cone"]
        """

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': instruction}
                ],
                options={
                    'temperature': 0.0, 
                    'num_ctx': 4096     
                },
                format='json'
            )
            
            content = response['message']['content']
            # 清理 Markdown
            content = re.sub(r'```json|```', '', content).strip()
            
            # 解析 JSON
            try:
                raw_data = json.loads(content)
            except json.JSONDecodeError:
                print(f"  [调试] JSON解析失败: {content}")
                return []

            # 智能提取列表 (修复 {"actions": ...} 的问题)
            final_list = self.extract_list_from_json(raw_data)
            
            return final_list

        except Exception as e:
            print(f"LLM Error: {e}")
            return []

if __name__ == "__main__":
    parser = NavigationLLM(model_name="qwen2.5:14b")

    dataset = [
        "move forward and stop at the tree",
        "move forward to the tree, turn right, go straight and stop at the traffic cone",
        "turn back, go straight to the tree, turn right, move until reach the bench",
        "head to your right hand side and go to the bench",
        "turn back, move forward to the tree, turn right, walk until you see a traffic cone",
        "move forward to the tractor trailer",
        "Walk straight to the billboard, turn right, and proceed until you reach a traffic cone",
        "Go straight ahead to the traffic cone, then turn left and continue to the billboard",
        "Turn left, walk straight to the billboard, turn left again, then move straight to the trash",
        "turn right, walk straight to the barrel, turn left and walk to the fire hydrant",
        "move forward to the traffic cone, turn right and continue to the billboard",
        "go forward to the tree",
        "walk forward to the billboard, then turn right, bypass pedestrians and continue to the traffic cone"
    ]

    print("\n" + "="*70)
    print(f"LLM ({parser.model}) 语义解析测试 (智能拆包版)")
    print("="*70)

    for i, cmd in enumerate(dataset, 1):
        print(f"\n[指令 {i}]: \"{cmd}\"")
        result = parser.parse_instruction(cmd)
        
        if result:
            is_valid_pair = (len(result) % 2 == 0) and (len(result) > 0)
            
            pairs_str = ""
            if is_valid_pair:
                pairs = [f"[{result[j]} -> {result[j+1]}]" for j in range(0, len(result), 2)]
                pairs_str = "  ".join(pairs)
            else:
                pairs_str = "❌ 格式错误 (非成对)"

            print(f" -> 原始数据: {json.dumps(result, ensure_ascii=False)}")
            print(f" -> 逻辑解析: {pairs_str}")
        else:
            print(" -> 解析失败 (空结果)")
    
    print("\n" + "="*70)