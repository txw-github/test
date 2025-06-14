
import re
import json
import os
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class TextPostProcessor:
    """文本后处理器 - 处理专业名词、多音字等识别错误"""
    
    def __init__(self, config_path: str = "text_correction_config.json"):
        self.config_path = config_path
        self.load_correction_rules()
        
    def load_correction_rules(self):
        """加载纠错规则"""
        default_rules = {
            # 专业名词纠错字典
            "professional_terms": {
                # 医学类
                "冠心病": ["关心病", "管心病"],
                "糖尿病": ["糖料病", "堂尿病"],
                "高血压": ["高血雅", "高血鸭"],
                "心脏病": ["心装病", "心藏病"],
                
                # 科技类
                "人工智能": ["人工只能", "人工职能"],
                "区块链": ["去块链", "区快链"],
                "大数据": ["大书据", "大数具"],
                "云计算": ["运计算", "云机算"],
                "物联网": ["物连网", "无联网"],
                
                # 法律类
                "合同": ["和同", "河同"],
                "诉讼": ["苏讼", "诉送"],
                "法院": ["法源", "发院"],
                "律师": ["绿师", "率师"],
                
                # 金融类
                "投资": ["头资", "投子"],
                "股票": ["骨票", "古票"],
                "基金": ["鸡金", "机金"],
                "保险": ["保现", "宝险"]
            },
            
            # 多音字纠错
            "polyphone_corrections": {
                # 根据上下文纠正多音字
                "银行": {
                    "错误": ["银航", "银杭"],
                    "关键词": ["金融", "存款", "贷款", "账户"]
                },
                "音乐": {
                    "错误": ["音月", "阴乐"],
                    "关键词": ["歌曲", "演出", "乐器", "唱歌"]
                },
                "重要": {
                    "错误": ["种要", "中要"],
                    "关键词": ["重点", "主要", "关键"]
                },
                "背景": {
                    "错误": ["被景", "倍景"],
                    "关键词": ["环境", "情况", "历史"]
                }
            },
            
            # 常见同音字错误
            "homophone_errors": {
                "的": ["得", "地"],
                "在": ["再"],
                "做": ["作"],
                "它": ["他", "她"],
                "像": ["象"],
                "以": ["已", "意"],
                "和": ["河", "何"],
                "或": ["货", "活"],
                "等": ["等"],
                "把": ["吧"],
                "被": ["背"],
                "没": ["美", "没"],
                "有": ["又", "友"]
            },
            
            # 数字和单位纠错
            "number_unit_corrections": {
                "十万": ["是万", "时万"],
                "百万": ["白万", "拜万"],
                "千万": ["钱万", "前万"],
                "公里": ["公立", "公力"],
                "公斤": ["公金", "公今"],
                "平方米": ["平方迷", "平方米"],
                "立方米": ["立方迷", "例方米"]
            }
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.rules = json.load(f)
                    # 合并默认规则
                    for key, value in default_rules.items():
                        if key not in self.rules:
                            self.rules[key] = value
                        else:
                            self.rules[key].update(value)
            else:
                self.rules = default_rules
                self.save_correction_rules()
        except Exception as e:
            logger.warning(f"加载纠错规则失败，使用默认规则: {e}")
            self.rules = default_rules
    
    def save_correction_rules(self):
        """保存纠错规则"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.rules, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存纠错规则失败: {e}")
    
    def correct_professional_terms(self, text: str) -> str:
        """纠正专业名词"""
        corrected_text = text
        
        for correct_term, wrong_terms in self.rules["professional_terms"].items():
            for wrong_term in wrong_terms:
                corrected_text = corrected_text.replace(wrong_term, correct_term)
        
        return corrected_text
    
    def correct_polyphone_by_context(self, text: str) -> str:
        """根据上下文纠正多音字"""
        corrected_text = text
        
        for correct_word, info in self.rules["polyphone_corrections"].items():
            wrong_words = info["错误"]
            keywords = info["关键词"]
            
            # 检查是否有相关关键词出现在文本中
            has_context = any(keyword in text for keyword in keywords)
            
            if has_context:
                for wrong_word in wrong_words:
                    corrected_text = corrected_text.replace(wrong_word, correct_word)
        
        return corrected_text
    
    def correct_numbers_and_units(self, text: str) -> str:
        """纠正数字和单位"""
        corrected_text = text
        
        for correct, wrong_list in self.rules["number_unit_corrections"].items():
            for wrong in wrong_list:
                corrected_text = corrected_text.replace(wrong, correct)
        
        # 数字格式化
        corrected_text = self._format_numbers(corrected_text)
        
        return corrected_text
    
    def _format_numbers(self, text: str) -> str:
        """格式化数字表达"""
        # 将中文数字转换为阿拉伯数字（简单处理）
        chinese_to_arabic = {
            '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
            '六': '6', '七': '7', '八': '8', '九': '9', '零': '0',
            '十': '10', '百': '100', '千': '1000', '万': '10000'
        }
        
        # 处理简单的中文数字
        for chinese, arabic in chinese_to_arabic.items():
            # 只替换独立的数字，避免误替换
            text = re.sub(f'(?<![\\u4e00-\\u9fff]){chinese}(?![\\u4e00-\\u9fff])', arabic, text)
        
        return text
    
    def correct_punctuation(self, text: str) -> str:
        """纠正标点符号"""
        # 移除多余的空格
        text = re.sub(r'\s+', ' ', text)
        
        # 句号前不应有空格
        text = re.sub(r'\s+。', '。', text)
        text = re.sub(r'\s+，', '，', text)
        text = re.sub(r'\s+？', '？', text)
        text = re.sub(r'\s+！', '！', text)
        
        # 添加适当的标点符号
        sentences = text.split('。')
        corrected_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and not sentence.endswith(('？', '！', '，')):
                if len(sentence) > 10:  # 较长的句子加句号
                    sentence += '。'
            corrected_sentences.append(sentence)
        
        return ''.join(corrected_sentences)
    
    def post_process(self, text: str, enable_all: bool = True) -> str:
        """执行完整的后处理"""
        logger.info("开始文本后处理...")
        
        original_text = text
        processed_text = text
        
        if enable_all:
            # 1. 专业名词纠错
            processed_text = self.correct_professional_terms(processed_text)
            
            # 2. 多音字上下文纠错
            processed_text = self.correct_polyphone_by_context(processed_text)
            
            # 3. 数字和单位纠错
            processed_text = self.correct_numbers_and_units(processed_text)
            
            # 4. 标点符号整理
            processed_text = self.correct_punctuation(processed_text)
        
        # 统计修改
        changes = self._count_changes(original_text, processed_text)
        if changes > 0:
            logger.info(f"文本后处理完成，共修正 {changes} 处错误")
        else:
            logger.info("文本后处理完成，未发现需要修正的错误")
        
        return processed_text
    
    def _count_changes(self, original: str, processed: str) -> int:
        """统计文本变化数量"""
        original_words = list(original)
        processed_words = list(processed)
        
        changes = 0
        min_len = min(len(original_words), len(processed_words))
        
        for i in range(min_len):
            if original_words[i] != processed_words[i]:
                changes += 1
        
        # 考虑长度差异
        changes += abs(len(original_words) - len(processed_words))
        
        return changes
    
    def add_custom_term(self, correct_term: str, wrong_terms: List[str]):
        """添加自定义专业名词"""
        if correct_term not in self.rules["professional_terms"]:
            self.rules["professional_terms"][correct_term] = []
        
        self.rules["professional_terms"][correct_term].extend(wrong_terms)
        self.save_correction_rules()
        logger.info(f"添加自定义词汇: {correct_term} <- {wrong_terms}")
    
    def get_correction_stats(self, text: str) -> Dict[str, int]:
        """获取纠错统计信息"""
        stats = {
            "professional_terms": 0,
            "polyphone_errors": 0,
            "number_units": 0,
            "total_chars": len(text)
        }
        
        # 统计专业名词错误
        for correct_term, wrong_terms in self.rules["professional_terms"].items():
            for wrong_term in wrong_terms:
                stats["professional_terms"] += text.count(wrong_term)
        
        # 统计多音字错误
        for correct_word, info in self.rules["polyphone_corrections"].items():
            for wrong_word in info["错误"]:
                stats["polyphone_errors"] += text.count(wrong_word)
        
        # 统计数字单位错误
        for correct, wrong_list in self.rules["number_unit_corrections"].items():
            for wrong in wrong_list:
                stats["number_units"] += text.count(wrong)
        
        return stats
