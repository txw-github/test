
import re
import json
import os
import jieba
import zhon.hanzi
from typing import Dict, List, Tuple, Optional, Set
import logging
from collections import defaultdict, Counter
import difflib

logger = logging.getLogger(__name__)

class EnhancedTextPostProcessor:
    """增强版文本后处理器 - 专业名词、多音字、错别字深度优化"""

    def __init__(self, config_file: str = "text_correction_config.json"):
        self.config_file = config_file
        self.load_config()
        
        # 初始化结巴分词
        jieba.initialize()
        
        # 专业词典缓存
        self.professional_cache = {}
        self.context_cache = {}
        
        # 加载增强词典
        self._load_enhanced_dictionaries()
        
        # 智能学习模块
        self.correction_statistics = defaultdict(int)
        self.context_patterns = defaultdict(list)

    def load_config(self):
        """加载增强配置信息"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = self._create_default_config()
            self.save_config()
            logger.warning(f"配置文件 {self.config_file} 未找到，已创建默认配置")
        except json.JSONDecodeError:
            self.config = self._create_default_config()
            logger.error(f"配置文件 {self.config_file} JSON格式错误，使用默认配置")

    def _create_default_config(self) -> Dict:
        """创建默认配置"""
        return {
            "professional_terms": {
                # 科技类
                "人工智能": ["人工只能", "人工职能", "人工智囊", "人攻智能"],
                "机器学习": ["机器雪习", "机器学西", "机器学细", "机器学席"],
                "深度学习": ["深度雪习", "深度学细", "申度学习", "深毒学习"],
                "神经网络": ["神经网洛", "神经忘络", "申经网络", "神精网络"],
                "算法": ["算发", "蒜法", "算花", "算华"],
                "数据库": ["数据哭", "数据苦", "属据库", "数据裤"],
                
                # 影视类
                "导演": ["道演", "倒演", "导言", "导眼"],
                "编剧": ["编据", "编局", "编剧", "编举"],
                "演员": ["言员", "眼员", "演圆", "眼原"],
                "制片": ["制篇", "制片", "只片", "制偏"],
                "剧情": ["据情", "举情", "剧青", "局情"],
                "角色": ["脚色", "叫色", "角涩", "交色"],
                
                # 医疗类
                "医院": ["医源", "一院", "衣院", "医原"],
                "医生": ["一生", "衣生", "医升", "易生"],
                "护士": ["护是", "护时", "护事", "户士"],
                "手术": ["手数", "首术", "收术", "守术"],
                "病人": ["病任", "病仁", "病认", "兵人"],
                
                # 商业类
                "公司": ["公思", "工司", "公司", "恭司"],
                "经理": ["经里", "京理", "经礼", "经力"],
                "客户": ["课户", "客服", "刻户", "客互"],
                "合作": ["和作", "合做", "河作", "核作"],
                "投资": ["头资", "投姿", "投子", "偷资"]
            },
            
            "polyphone_corrections": {
                "银行": {
                    "错误": ["银航", "银杭", "音行", "阴行", "印行"],
                    "关键词": ["金融", "存款", "贷款", "账户", "转账", "利息", "取钱", "ATM", "卡号"]
                },
                "音乐": {
                    "错误": ["音月", "阴乐", "因乐", "印乐"],
                    "关键词": ["歌曲", "演出", "乐器", "唱歌", "音符", "旋律", "节拍", "歌手"]
                },
                "重要": {
                    "错误": ["种要", "中要", "重药", "众要"],
                    "关键词": ["重点", "主要", "关键", "核心", "重大", "重视", "紧要"]
                },
                "背景": {
                    "错误": ["被景", "倍景", "背井", "背经"],
                    "关键词": ["环境", "情况", "历史", "经历", "出身", "背后", "幕后"]
                },
                "教育": {
                    "错误": ["叫育", "交育", "教鱼", "焦育"],
                    "关键词": ["学校", "老师", "学生", "课程", "学习", "培养", "教学"]
                },
                "干净": {
                    "错误": ["干经", "甘净", "干境", "赶净"],
                    "关键词": ["清洁", "整洁", "卫生", "洗", "擦", "打扫"]
                },
                "调查": {
                    "错误": ["条查", "调茶", "掉查", "刁查"],
                    "关键词": ["研究", "调研", "调查", "问卷", "统计", "分析"]
                }
            },
            
            "context_corrections": {
                "的地得": {
                    "的": {
                        "pattern": r"(\w+)地(\w+)",
                        "conditions": ["名词", "代词"],
                        "examples": ["我的书", "他的家"]
                    },
                    "地": {
                        "pattern": r"(\w+)的(\w+地)$",
                        "conditions": ["副词", "形容词"],
                        "examples": ["快速地跑", "认真地做"]
                    },
                    "得": {
                        "pattern": r"(\w+)的(\w+)$",
                        "conditions": ["动词", "形容词"],
                        "examples": ["跑得快", "做得好"]
                    }
                }
            },
            
            "advanced_corrections": {
                "enable_context_analysis": True,
                "enable_frequency_analysis": True,
                "enable_semantic_analysis": True,
                "learning_mode": True
            }
        }

    def save_config(self):
        """保存配置"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"配置保存失败: {e}")

    def _load_enhanced_dictionaries(self):
        """加载增强词典"""
        # 常见同音字错误扩展
        self.enhanced_homophones = {
            # 基础同音字
            "在": ["再", "载"],
            "再": ["在", "载"],
            "做": ["作", "坐"],
            "作": ["做", "坐"],
            "座": ["坐", "做"],
            
            # 多音字常见错误
            "了": ["理", "里", "李"],
            "要": ["药", "腰", "摇"],
            "好": ["号", "毫", "豪"],
            "会": ["汇", "慧", "惠"],
            "得": ["的", "地", "德"],
            
            # 电视剧常见词汇
            "演员": ["言员", "眼员", "演圆"],
            "导演": ["道演", "倒演", "导言"],
            "剧情": ["据情", "举情", "剧青"],
            "角色": ["脚色", "叫色", "交色"],
            "情节": ["情结", "青洁", "清洁"],
            
            # 现代词汇
            "网络": ["网洛", "忘络", "网络"],
            "手机": ["手鸡", "收机", "守机"],
            "电脑": ["电闹", "点脑", "电瑙"],
            "软件": ["软键", "软见", "软建"],
            
            # 方位时间
            "这里": ["这礼", "这理", "这李"],
            "那里": ["那礼", "那理", "那李"],
            "现在": ["县在", "现再", "线在"],
            "刚才": ["刚财", "钢才", "刚材"],
            "以后": ["以厚", "已后", "一后"],
            "以前": ["以钱", "已前", "一前"],
            
            # 情感表达
            "喜欢": ["洗欢", "喜换", "系欢"],
            "讨厌": ["淘嫌", "讨险", "逃嫌"],
            "高兴": ["高性", "告兴", "搞兴"],
            "生气": ["声气", "升气", "圣气"],
            "难过": ["南过", "男过", "难国"],
            
            # 日常用语
            "吃饭": ["齿饭", "吃反", "迟饭"],
            "睡觉": ["水觉", "睡脚", "税觉"],
            "工作": ["攻作", "公作", "功作"],
            "学习": ["雪习", "学西", "学细"],
            "买东西": ["买冬西", "买东细", "迈东西"]
        }
        
        # 专业领域词典
        self.domain_terms = {
            "technology": ["人工智能", "机器学习", "深度学习", "神经网络", "算法", "数据库"],
            "entertainment": ["导演", "编剧", "演员", "制片", "剧情", "角色", "情节"],
            "medical": ["医院", "医生", "护士", "手术", "病人", "治疗", "诊断"],
            "business": ["公司", "经理", "客户", "合作", "投资", "营销", "管理"],
            "education": ["学校", "老师", "学生", "课程", "教育", "培训", "考试"]
        }

    def _analyze_context(self, text: str, target_word: str, position: int) -> Dict:
        """分析上下文信息"""
        try:
            # 获取目标词前后的上下文
            words = list(jieba.cut(text))
            
            # 找到目标词位置
            word_positions = []
            current_pos = 0
            for i, word in enumerate(words):
                if current_pos <= position < current_pos + len(word):
                    word_positions.append(i)
                current_pos += len(word)
            
            if not word_positions:
                return {}
            
            word_idx = word_positions[0]
            
            # 提取上下文特征
            context = {
                "prev_words": words[max(0, word_idx-3):word_idx],
                "next_words": words[word_idx+1:min(len(words), word_idx+4)],
                "sentence": text,
                "domain": self._detect_domain(words),
                "pos_tag": self._simple_pos_tag(target_word, words, word_idx)
            }
            
            return context
            
        except Exception as e:
            logger.debug(f"上下文分析失败: {e}")
            return {}

    def _detect_domain(self, words: List[str]) -> str:
        """检测文本领域"""
        domain_scores = defaultdict(int)
        
        for word in words:
            for domain, terms in self.domain_terms.items():
                if word in terms:
                    domain_scores[domain] += 1
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        return "general"

    def _simple_pos_tag(self, word: str, words: List[str], position: int) -> str:
        """简单词性标注"""
        # 基于位置和上下文的简单词性判断
        if position == 0:
            return "sentence_start"
        
        prev_word = words[position - 1] if position > 0 else ""
        next_word = words[position + 1] if position < len(words) - 1 else ""
        
        # 简单规则
        if prev_word in ["的", "在", "从", "到"]:
            return "noun"
        elif next_word in ["的", "了", "着", "过"]:
            return "verb"
        elif word.endswith(("地", "得")):
            return "adverb"
        else:
            return "unknown"

    def _smart_homophone_correction(self, text: str) -> str:
        """智能同音字纠错"""
        corrected_text = text
        corrections_made = []
        
        # 按词长度排序，优先处理长词
        all_corrections = {}
        all_corrections.update(self.enhanced_homophones)
        all_corrections.update({k: v for k, v in self.config.get("professional_terms", {}).items()})
        
        # 按长度排序
        sorted_terms = sorted(all_corrections.items(), key=lambda x: len(x[0]), reverse=True)
        
        for correct_word, wrong_words in sorted_terms:
            for wrong_word in wrong_words:
                if wrong_word in corrected_text:
                    # 上下文验证
                    positions = [m.start() for m in re.finditer(re.escape(wrong_word), corrected_text)]
                    
                    for pos in reversed(positions):  # 从后往前替换，避免位置偏移
                        context = self._analyze_context(corrected_text, wrong_word, pos)
                        
                        # 决定是否替换
                        should_replace = self._should_replace_word(correct_word, wrong_word, context)
                        
                        if should_replace:
                            corrected_text = (corrected_text[:pos] + 
                                            correct_word + 
                                            corrected_text[pos + len(wrong_word):])
                            corrections_made.append((wrong_word, correct_word, pos))
        
        if corrections_made:
            logger.debug(f"智能同音字纠错: {len(corrections_made)} 处修改")
            
        return corrected_text

    def _should_replace_word(self, correct_word: str, wrong_word: str, context: Dict) -> bool:
        """判断是否应该替换词汇"""
        # 基础相似度检查
        similarity = difflib.SequenceMatcher(None, correct_word, wrong_word).ratio()
        if similarity < 0.3:  # 相似度太低，可能不是同音字错误
            return False
        
        # 领域匹配检查
        domain = context.get("domain", "general")
        if domain != "general":
            # 如果是专业领域，优先使用专业词汇
            if correct_word in self.domain_terms.get(domain, []):
                return True
        
        # 上下文关键词检查
        if correct_word in self.config.get("polyphone_corrections", {}):
            poly_info = self.config["polyphone_corrections"][correct_word]
            keywords = poly_info.get("关键词", [])
            
            # 检查上下文中是否包含关键词
            context_text = " ".join(context.get("prev_words", []) + context.get("next_words", []))
            if any(keyword in context_text for keyword in keywords):
                return True
        
        # 默认替换策略
        return True

    def _advanced_punctuation_processing(self, text: str) -> str:
        """高级标点符号处理"""
        if not text:
            return text
        
        # 使用jieba分词获得更好的断句
        words = list(jieba.cut(text))
        
        # 语义单元识别
        semantic_units = self._identify_semantic_units(words)
        
        # 重新组装带标点的文本
        result_parts = []
        
        for i, unit in enumerate(semantic_units):
            unit_text = "".join(unit["words"])
            unit_type = unit["type"]
            
            # 根据语义单元类型添加标点
            if unit_type == "question":
                if not unit_text.endswith("？"):
                    unit_text += "？"
            elif unit_type == "exclamation":
                if not unit_text.endswith("！"):
                    unit_text += "！"
            elif unit_type == "statement":
                if i == len(semantic_units) - 1:  # 最后一个单元
                    if not unit_text.endswith(("。", "！", "？")):
                        unit_text += "。"
                else:
                    # 中间单元，根据长度决定标点
                    if len(unit_text) > 15 and not unit_text.endswith(("，", "。", "！", "？")):
                        unit_text += "，"
            
            result_parts.append(unit_text)
        
        result = "".join(result_parts)
        
        # 后处理：清理多余标点
        result = re.sub(r'[，]{2,}', '，', result)
        result = re.sub(r'[。]{2,}', '。', result)
        result = re.sub(r'[！]{2,}', '！', result)
        result = re.sub(r'[？]{2,}', '？', result)
        
        return result

    def _identify_semantic_units(self, words: List[str]) -> List[Dict]:
        """识别语义单元"""
        units = []
        current_unit = {"words": [], "type": "statement"}
        
        question_indicators = ["什么", "哪里", "怎么", "为什么", "如何", "多少", "几", "吗", "呢"]
        exclamation_indicators = ["太", "真", "好", "很", "非常", "特别", "哇", "啊"]
        transition_words = ["但是", "不过", "然而", "可是", "而且", "并且", "然后", "接着", "于是", "所以"]
        
        for i, word in enumerate(words):
            current_unit["words"].append(word)
            
            # 检测疑问句
            if any(indicator in word for indicator in question_indicators):
                current_unit["type"] = "question"
            
            # 检测感叹句
            elif any(indicator in word for indicator in exclamation_indicators):
                current_unit["type"] = "exclamation"
            
            # 句子边界检测
            if (word in transition_words or 
                len(current_unit["words"]) > 20 or 
                (i < len(words) - 1 and words[i+1] in transition_words)):
                
                if current_unit["words"]:
                    units.append(current_unit)
                    current_unit = {"words": [], "type": "statement"}
        
        # 添加最后一个单元
        if current_unit["words"]:
            units.append(current_unit)
        
        return units

    def _enhance_professional_terms(self, text: str) -> str:
        """增强专业术语处理"""
        corrected_text = text
        
        # 获取所有专业术语
        professional_terms = self.config.get("professional_terms", {})
        
        # 按词长排序，先处理长词
        sorted_terms = sorted(professional_terms.items(), key=lambda x: len(x[0]), reverse=True)
        
        for correct_term, wrong_terms in sorted_terms:
            for wrong_term in wrong_terms:
                if wrong_term in corrected_text:
                    # 上下文检查
                    context_valid = True
                    
                    # 如果有特定的上下文要求，进行验证
                    if correct_term in self.config.get("polyphone_corrections", {}):
                        poly_info = self.config["polyphone_corrections"][correct_term]
                        keywords = poly_info.get("关键词", [])
                        
                        if keywords:
                            context_valid = any(keyword in corrected_text for keyword in keywords)
                    
                    if context_valid:
                        corrected_text = corrected_text.replace(wrong_term, correct_term)
                        logger.debug(f"专业术语纠正: {wrong_term} -> {correct_term}")
        
        return corrected_text

    def _statistical_learning(self, original_text: str, corrected_text: str):
        """统计学习模块"""
        if not self.config.get("advanced_corrections", {}).get("learning_mode", False):
            return
        
        try:
            # 记录纠错统计
            if original_text != corrected_text:
                self.correction_statistics["total_corrections"] += 1
                
                # 分析修改类型
                changes = self._analyze_changes(original_text, corrected_text)
                for change_type in changes:
                    self.correction_statistics[change_type] += 1
                
                # 保存学习结果
                self._save_learning_data()
                
        except Exception as e:
            logger.debug(f"统计学习失败: {e}")

    def _analyze_changes(self, original: str, corrected: str) -> List[str]:
        """分析文本变化类型"""
        changes = []
        
        # 检测标点符号变化
        original_puncts = len(re.findall(r'[，。！？；：]', original))
        corrected_puncts = len(re.findall(r'[，。！？；：]', corrected))
        if corrected_puncts > original_puncts:
            changes.append("punctuation_added")
        
        # 检测同音字纠错
        for correct, wrongs in self.enhanced_homophones.items():
            for wrong in wrongs:
                if wrong in original and correct in corrected:
                    changes.append("homophone_correction")
                    break
        
        # 检测专业术语纠错
        for correct, wrongs in self.config.get("professional_terms", {}).items():
            for wrong in wrongs:
                if wrong in original and correct in corrected:
                    changes.append("professional_term_correction")
                    break
        
        return changes

    def _save_learning_data(self):
        """保存学习数据"""
        try:
            learning_file = "learning_data.json"
            learning_data = {
                "statistics": dict(self.correction_statistics),
                "timestamp": str(time.time())
            }
            
            with open(learning_file, 'w', encoding='utf-8') as f:
                json.dump(learning_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.debug(f"学习数据保存失败: {e}")

    def post_process(self, text: str) -> str:
        """主要的文本后处理方法 - 增强版"""
        if not text or not text.strip():
            return text

        original_text = text
        
        try:
            # 第1步: 基础清理
            text = self._clean_text(text)
            
            # 第2步: 智能同音字纠错
            text = self._smart_homophone_correction(text)
            
            # 第3步: 增强专业术语处理
            text = self._enhance_professional_terms(text)
            
            # 第4步: 上下文相关的多音字纠正
            text = self._context_aware_polyphone_correction(text)
            
            # 第5步: 数字和单位处理
            text = self._process_numbers_and_units(text)
            
            # 第6步: 语气词和填充词处理
            text = self._correct_filler_words(text)
            
            # 第7步: 高级标点符号处理
            text = self._advanced_punctuation_processing(text)
            
            # 第8步: 最终优化
            text = self._final_optimization(text)
            
            # 统计学习
            self._statistical_learning(original_text, text)
            
            # 统计修改情况
            changes = self._count_changes(original_text, text)
            if changes > 0:
                logger.info(f"✨ 文本后处理完成，共优化 {changes} 处")
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"❌ 文本后处理失败: {e}")
            return original_text

    def _context_aware_polyphone_correction(self, text: str) -> str:
        """上下文感知的多音字纠正"""
        corrected_text = text
        polyphone_corrections = self.config.get("polyphone_corrections", {})
        
        for correct_word, context_info in polyphone_corrections.items():
            # 兼容新旧格式
            if "错误" in context_info:
                incorrect_words = context_info["错误"]
                keywords = context_info.get("关键词", [])
            else:
                incorrect_words = context_info.get("incorrect", [])
                keywords = context_info.get("keywords", [])
            
            # 检查关键词上下文
            has_context = any(keyword in text for keyword in keywords)
            
            if has_context:
                for incorrect_word in incorrect_words:
                    if incorrect_word in corrected_text:
                        # 额外的上下文验证
                        positions = [m.start() for m in re.finditer(re.escape(incorrect_word), corrected_text)]
                        
                        for pos in reversed(positions):
                            context = self._analyze_context(corrected_text, incorrect_word, pos)
                            
                            # 更精确的上下文匹配
                            local_context = " ".join(context.get("prev_words", []) + context.get("next_words", []))
                            local_keywords = [kw for kw in keywords if kw in local_context]
                            
                            if local_keywords:
                                corrected_text = (corrected_text[:pos] + 
                                                correct_word + 
                                                corrected_text[pos + len(incorrect_word):])
                                logger.debug(f"上下文多音字纠正: {incorrect_word} -> {correct_word} (关键词: {local_keywords})")
        
        return corrected_text

    def _clean_text(self, text: str) -> str:
        """增强的基础文本清理"""
        text = text.strip()
        
        # 合并多个空格为一个
        text = re.sub(r'\s+', ' ', text)
        
        # 移除多余的换行符
        text = re.sub(r'\n+', ' ', text)
        
        # 修复常见的OCR错误
        ocr_fixes = {
            "0": "。",  # 数字0误识别为句号
            "1": "！",  # 数字1误识别为感叹号
            "|": "！",  # 竖线误识别为感叹号
            "？？": "？",
            "！！": "！",
            "。。": "。"
        }
        
        for wrong, correct in ocr_fixes.items():
            text = text.replace(wrong, correct)
        
        return text

    def _correct_filler_words(self, text: str) -> str:
        """增强的语气词和填充词处理"""
        # 基础填充词纠正
        filler_corrections = {
            "嗯嗯": "嗯",
            "额额": "额", 
            "呃呃": "呃",
            "哦哦": "哦",
            "啊啊": "啊",
            "呵呵呵": "呵呵",
            "哈哈哈": "哈哈",
            "嘿嘿嘿": "嘿嘿",
            "就是说": "就是说",
            "然后呢": "然后",
            "那个那个": "那个",
            "这个这个": "这个",
            "对对对": "对",
            "是是是": "是",
            "好好好": "好",
            "行行行": "行"
        }
        
        for correct, wrong in filler_corrections.items():
            text = text.replace(wrong, correct)
        
        # 处理重复的语气词
        text = re.sub(r'([啊哦嗯额呃哈呵嘿])\1{2,}', r'\1', text)
        
        # 处理重复的字词（超过2次重复）
        text = re.sub(r'(.)\1{3,}', r'\1\1', text)
        
        # 移除无意义的填充词
        meaningless_fillers = ["那个", "这个", "就是", "然后", "嗯", "额"]
        words = list(jieba.cut(text))
        
        # 智能移除：只在密度过高时移除
        filler_count = sum(1 for word in words if word in meaningless_fillers)
        if len(words) > 0 and filler_count / len(words) > 0.3:  # 填充词占比超过30%
            # 保留关键的填充词，移除多余的
            cleaned_words = []
            prev_was_filler = False
            
            for word in words:
                if word in meaningless_fillers:
                    if not prev_was_filler:  # 避免连续的填充词
                        cleaned_words.append(word)
                        prev_was_filler = True
                    # 否则跳过
                else:
                    cleaned_words.append(word)
                    prev_was_filler = False
            
            text = "".join(cleaned_words)
        
        return text

    def _process_numbers_and_units(self, text: str) -> str:
        """增强的数字和单位处理"""
        # 基础数字单位纠正
        number_unit_corrections = self.config.get("number_unit_corrections", {})
        
        for correct_form, incorrect_forms in number_unit_corrections.items():
            for incorrect_form in incorrect_forms:
                text = text.replace(incorrect_form, correct_form)
        
        # 数字读音纠正 - 更全面
        digit_corrections = {
            "零": ["另", "令", "灵"],
            "一": ["医", "衣", "依"],
            "二": ["尔", "而", "耳"],
            "三": ["伞", "散", "山"],
            "四": ["是", "死", "司"],
            "五": ["我", "吴", "午"],
            "六": ["留", "流", "柳"],
            "七": ["期", "齐", "奇"],
            "八": ["吧", "把", "爸"],
            "九": ["就", "酒", "久"],
            "十": ["是", "实", "时"]
        }
        
        for correct, wrong_list in digit_corrections.items():
            for wrong in wrong_list:
                # 只在数字上下文中替换
                text = re.sub(f'({wrong})([十百千万亿])', f'{correct}\\2', text)
                text = re.sub(f'([十百千万亿])({wrong})', f'\\1{correct}', text)
                
                # 处理纯数字
                text = re.sub(f'第({wrong})', f'第{correct}', text)
                text = re.sub(f'({wrong})个', f'{correct}个', text)
        
        # 处理年份、时间等特殊数字
        # 例：二零二三年 -> 2023年
        year_pattern = r'([一二三四五六七八九零〇]{4})年'
        def convert_chinese_year(match):
            chinese_digits = {'零': '0', '〇': '0', '一': '1', '二': '2', '三': '3', '四': '4', 
                            '五': '5', '六': '6', '七': '7', '八': '8', '九': '9'}
            year_str = match.group(1)
            converted = ''.join(chinese_digits.get(char, char) for char in year_str)
            return f'{converted}年'
        
        text = re.sub(year_pattern, convert_chinese_year, text)
        
        return text

    def _final_optimization(self, text: str) -> str:
        """最终优化"""
        # 移除多余的空格
        text = re.sub(r'\s+', '', text)
        
        # 处理连续的标点符号
        text = re.sub(r'[，]{2,}', '，', text)
        text = re.sub(r'[。]{2,}', '。', text)
        text = re.sub(r'[！]{2,}', '！', text)
        text = re.sub(r'[？]{2,}', '？', text)
        
        # 处理标点符号前的空格
        text = re.sub(r'\s+([，。！？；：])', r'\1', text)
        
        # 智能标点符号决策
        if text and not text.endswith(('。', '！', '？')):
            # 根据内容决定结尾标点
            if any(qw in text[-20:] for qw in ['什么', '怎么', '为什么', '哪里', '吗', '呢']):
                text += '？'
            elif any(ew in text[-20:] for ew in ['太', '真', '好', '很', '非常', '哇', '啊']):
                text += '！'
            else:
                text += '。'
        
        return text

    def _count_changes(self, original: str, processed: str) -> int:
        """统计文本变化数量"""
        if len(original) == 0 and len(processed) == 0:
            return 0
        
        # 使用编辑距离计算变化
        import difflib
        changes = 0
        
        # 计算字符级别的差异
        for op in difflib.ndiff(original, processed):
            if op.startswith('+ ') or op.startswith('- '):
                changes += 1
        
        return changes // 2  # 每个替换会产生一个删除和一个添加

    def get_correction_stats(self, text: str) -> Dict[str, int]:
        """获取详细的纠错统计信息"""
        stats = {
            "sound_alike_errors": 0,
            "professional_terms": 0, 
            "polyphone_errors": 0,
            "number_units": 0,
            "filler_words": 0,
            "punctuation_added": 0,
            "total_chars": len(text),
            "potential_improvements": 0
        }

        # 统计各类错误
        for correct, wrong_list in self.enhanced_homophones.items():
            for wrong in wrong_list:
                stats["sound_alike_errors"] += text.count(wrong)

        professional_terms = self.config.get("professional_terms", {})
        for correct_term, wrong_terms in professional_terms.items():
            for wrong_term in wrong_terms:
                stats["professional_terms"] += text.count(wrong_term)

        polyphone_corrections = self.config.get("polyphone_corrections", {})
        for correct_word, info in polyphone_corrections.items():
            wrong_words = info.get("incorrect", info.get("错误", []))
            for wrong_word in wrong_words:
                stats["polyphone_errors"] += text.count(wrong_word)

        # 估算可能的改进点
        stats["potential_improvements"] = (stats["sound_alike_errors"] + 
                                         stats["professional_terms"] + 
                                         stats["polyphone_errors"])

        return stats

    def add_custom_correction(self, correct_word: str, wrong_words: List[str], 
                            category: str = "custom", keywords: List[str] = None):
        """添加自定义纠错词汇"""
        if category == "sound_alike":
            self.enhanced_homophones[correct_word] = wrong_words
        elif category == "professional":
            if "professional_terms" not in self.config:
                self.config["professional_terms"] = {}
            self.config["professional_terms"][correct_word] = wrong_words
        elif category == "polyphone":
            if "polyphone_corrections" not in self.config:
                self.config["polyphone_corrections"] = {}
            self.config["polyphone_corrections"][correct_word] = {
                "错误": wrong_words,
                "关键词": keywords or []
            }
        
        # 保存配置
        self.save_config()
        logger.info(f"✅ 添加自定义纠错词汇 ({category}): {correct_word} <- {wrong_words}")

# 向后兼容
class TextPostProcessor(EnhancedTextPostProcessor):
    """向后兼容的文本后处理器"""
    pass
