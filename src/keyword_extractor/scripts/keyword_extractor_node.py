#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from keyword_extractor.msg import KeywordsResult

class KeywordExtractor:
    def __init__(self):
        rospy.init_node('keyword_extractor', anonymous=True, log_level=rospy.DEBUG)
        
        # 初始化发布器和订阅器
        self.publisher = rospy.Publisher('/extracted_keywords', KeywordsResult, queue_size=10, latch=True)
        rospy.Subscriber('/instruction', String, self.callback)
        
        # 定义关键词映射
        self.mapping = {
            "right": "right",
            "left": "left",
            "back": "back",
            "backward": "back",
            "front": "front",
            "forward": "front",
            "advance": "front",
            "straight": "front",
            "continue": "front",
            "proceed": "front",
            "ahead": "front",
            "until": "front",
            "traffic cone": "traffic cone",
            "the cone": "traffic cone",
            "a cone": "traffic cone",
            "trash": "trash",
            "bench": "bench",
            "billboard": "billboard",
            "tree": "tree",
            "tractor trailer": "tractor trailer",
            "barrel": "barrel",
            "fire hydrant": "fire hydrant",
            "pass": "pass",
            "bypass": "pass",
            "pedestrians": "human",
            "pedestrian": "human",
            "human": "human",
            "person": "human",
            "people": "human",
            "walker": "human",
            "bystander": "human",
            "passerby": "human",
            "man": "human"
        }
        
        # 创建关键词列表，按长度降序排列
        self.keywords = sorted(self.mapping.keys(), key=len, reverse=True)
        
        # 用于分类关键词的集合，便于逻辑处理
        self.directions = {"right", "left", "back", "front"}
        self.objects = {v for k, v in self.mapping.items() if v not in self.directions and v != "pass"}
        
        rospy.loginfo("Keyword Extractor 节点已启动，等待 /instruction 消息...")
        rospy.loginfo(f"可识别的关键词: {self.keywords}")
    
    def extract_keywords(self, s):
        """从字符串中提取关键词 - 使用精确匹配"""
        input_lower = s.lower()
        found_keywords = []
        i = 0
        n = len(input_lower)
        
        while i < n:
            matched = False
            for keyword in self.keywords:
                if input_lower.startswith(keyword, i):
                    start_ok = (i == 0 or not input_lower[i-1].isalnum())
                    end_pos = i + len(keyword)
                    end_ok = (end_pos >= n or not input_lower[end_pos].isalnum())
                    
                    if start_ok and end_ok:
                        found_keywords.append(self.mapping[keyword])
                        i = end_pos
                        matched = True
                        break
            if not matched:
                i += 1
        return found_keywords

    def _preprocess_front_commands(self, keywords):
        """
        第一优先级处理：只保留位于指令序列第一个的 'front'，移除所有后续的 'front'。
        """
        if not keywords:
            return []
        
        # 无条件保留第一个关键词
        processed_keywords = [keywords[0]]
        # 从第二个关键词开始遍历
        for i in range(1, len(keywords)):
            # 只要不是 'front'，就予以保留
            if keywords[i] != "front":
                processed_keywords.append(keywords[i])
        
        rospy.loginfo(f"预处理 'front' 后: {processed_keywords}")
        return processed_keywords

    def _handle_pass_command(self, keywords):
        """
        第二优先级处理：如果 'pass' 后面跟着一个物体，则将两者都移除。
        """
        processed_keywords = []
        i = 0
        while i < len(keywords):
            current_keyword = keywords[i]
            if current_keyword == "pass" and (i + 1) < len(keywords) and keywords[i+1] in self.objects:
                rospy.logdebug(f"找到 'pass' 模式: 移除 '{current_keyword}' 和 '{keywords[i+1]}'")
                i += 2
                continue
            
            processed_keywords.append(current_keyword)
            i += 1
        
        rospy.loginfo(f"处理 'pass' 后: {processed_keywords}")
        return processed_keywords

    def _reorder_for_object_conditions(self, keywords):
        """
        第三优先级处理：为 "在物体处执行动作" 的情况重排序关键词。
        例如: ["left", "right", "tree"] -> ["left", "tree", "right"]
        """
        reordered_keywords = list(keywords)
        i = 0
        while i <= len(reordered_keywords) - 3:
            is_dir1 = reordered_keywords[i] in self.directions
            is_dir2 = reordered_keywords[i+1] in self.directions
            is_obj = reordered_keywords[i+2] in self.objects
            
            if is_dir1 and is_dir2 and is_obj:
                rospy.logdebug(f"找到模式: {reordered_keywords[i:i+3]}。交换 '{reordered_keywords[i+1]}' 和 '{reordered_keywords[i+2]}'")
                reordered_keywords[i+1], reordered_keywords[i+2] = reordered_keywords[i+2], reordered_keywords[i+1]
                i += 2
            else:
                i += 1
        
        rospy.loginfo(f"重排序后: {reordered_keywords}")
        return reordered_keywords

    def callback(self, msg):
        """处理接收到的消息"""
        rospy.loginfo(f"收到指令: {msg.data}")
        
        try:
            # 步骤1: 按出现顺序提取关键词
            initial_keywords = self.extract_keywords(msg.data)
            rospy.loginfo(f"原始关键词: {initial_keywords}")
            
            # 步骤2: (最高优先级) 预处理 "front" 指令
            front_processed_keywords = self._preprocess_front_commands(initial_keywords)
            
            # 步骤3: 处理 "pass" 指令
            pass_handled_keywords = self._handle_pass_command(front_processed_keywords)
            
            # 步骤4: 对清理后的列表进行重排序
            final_keywords = self._reorder_for_object_conditions(pass_handled_keywords)
            
            rospy.loginfo(f"最终发布的关键词: {final_keywords}")
            
            # 步骤5: 创建并发布结果消息
            result_msg = KeywordsResult()
            result_msg.keywords = final_keywords
            self.publisher.publish(result_msg)
            
            rospy.loginfo(f"已发布到 /extracted_keywords")
        except Exception as e:
            rospy.logerr(f"处理消息时出错: {str(e)}")

if __name__ == '__main__':
    try:
        rospy.loginfo("启动关键词提取节点...")
        extractor = KeywordExtractor()
        rospy.loginfo("进入ROS循环...")
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("节点被中断")
    except Exception as e:
        rospy.logfatal(f"节点崩溃: {str(e)}")
