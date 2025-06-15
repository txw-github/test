
import re
import json
import os
import jieba
import zhon.hanzi
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TextPostProcessor:
    """文本后处理器 - 处理专业名词、多音字、错别字、智能断句等识别错误"""

    def __init__(self, config_file: str = "text_correction_config.json"):
        self.config_file = config_file
        self.load_config()

        # 初始化结巴分词
        jieba.initialize()

        # 常见音对字不对的错误词典
        self.sound_alike_errors = {
            # 语音识别常见的同音字错误
            "在哪里": ["在那里", "在纳里"],
            "怎么样": ["怎么羊", "怎么央"],
            "什么时候": ["什么事后", "什么时后"],
            "为什么": ["为甚么", "围什么"],
            "这样子": ["这羊子", "这样紫"],
            "那样子": ["那羊子", "那样紫"],
            "没关系": ["没观系", "没关细"],
            "不要紧": ["不要金", "不要斤"],
            "有意思": ["有意识", "有意丝"],
            "没意思": ["没意识", "没意丝"],
            "不好意思": ["不好意识", "不好意丝"],
            "一会儿": ["一会而", "一回儿"],
            "过一会儿": ["过一会而", "过一回儿"],
            "等一下": ["等以下", "等一夏"],
            "等一等": ["等一灯", "等一邓"],
            "再见": ["在见", "再间"],
            "拜拜": ["白白", "败败"],
            "谢谢": ["些些", "写写"],
            "对不起": ["对不齐", "对不起"],
            "麻烦": ["马烦", "妈烦"],
            "辛苦": ["心苦", "新苦"],
            "厉害": ["利害", "历害"],
            "表现": ["表县", "表显"],
            "表示": ["表是", "表适"],
            "注意": ["主意", "住意"],
            "小心": ["小新", "小信"],
            "当心": ["当新", "当信"],
            "担心": ["单心", "弹心"],
            "放心": ["方心", "防心"],
            "开心": ["开新", "开信"],
            "高兴": ["高性", "告兴"],
            "兴奋": ["性奋", "醒奋"],
            "激动": ["即动", "急动"],
            "紧张": ["金张", "近张"],
            "着急": ["着记", "着及"],
            "特别": ["特白", "特北"],
            "尤其": ["尤齐", "犹其"],
            "尤其是": ["尤齐是", "犹其是"],
            "关键": ["观键", "关剑"],
            "重要": ["种要", "中要"],
            "主要": ["主药", "珠要"],
            "首要": ["手要", "收要"],
            "必要": ["比要", "必药"],
            "需要": ["须要", "需药"],
            "一定": ["一顶", "一订"],
            "肯定": ["肯丁", "肯订"],
            "当然": ["当染", "党然"],
            "自然": ["自燃", "紫然"],
            "当时": ["当实", "当是"],
            "现在": ["县在", "现再"],
            "以前": ["以钱", "已前"],
            "以后": ["以厚", "已后"],
            "以来": ["以来", "已来"],
            "以上": ["以商", "已上"],
            "以下": ["以夏", "已下"],
        }

        # 电视剧常见专业词汇
        self.tv_drama_terms = {
            "主角": ["主脚", "朱角"],
            "配角": ["陪角", "配脚"],
            "演员": ["言员", "眼员"],
            "导演": ["道演", "倒演"],
            "编剧": ["编据", "编局"],
            "制片": ["制篇", "制片"],
            "剧本": ["据本", "剧奔"],
            "台词": ["台词", "抬词"],
            "对白": ["对百", "对摆"],
            "情节": ["情结", "清洁"],
            "剧情": ["据情", "举情"],
            "故事": ["古事", "股事"],
            "角色": ["脚色", "叫色"],
            "人物": ["人勿", "任物"],
            "性格": ["性各", "星格"],
            "感情": ["敢情", "感清"],
            "爱情": ["爱清", "哀情"],
            "友情": ["友清", "有情"],
            "亲情": ["亲清", "青情"],
            "婚姻": ["昏姻", "婚银"],
            "家庭": ["家廷", "家停"],
            "公司": ["公思", "工司"],
            "学校": ["学效", "雪校"],
            "医院": ["医源", "一院"],
            "警察": ["警擦", "警察"],
            "律师": ["绿师", "率师"],
            "医生": ["一生", "衣生"],
            "老师": ["老是", "老师"],
            "学生": ["雪生", "学声"],
            "朋友": ["朋有", "朋友"],
            "同事": ["同是", "童事"],
            "同学": ["同雪", "童学"],
            "邻居": ["邻据", "临居"],
            "客人": ["课人", "刻人"],
            "房子": ["房紫", "方子"],
            "房间": ["房键", "房建"],
            "客厅": ["课厅", "客停"],
            "卧室": ["我室", "卧是"],
            "厨房": ["除房", "厨方"],
            "浴室": ["欲室", "浴是"],
            "阳台": ["羊台", "杨台"],
            "花园": ["花元", "华园"],
            "车库": ["车苦", "车哭"],
            "办公室": ["办公是", "办工室"],
            "会议室": ["会议是", "回忆室"],
            "餐厅": ["餐停", "惨厅"],
            "酒店": ["酒店", "就店"],
            "宾馆": ["宾管", "滨馆"],
            "商场": ["伤场", "商厂"],
            "超市": ["超是", "朝市"],
            "银行": ["音行", "印行"],
            "机场": ["鸡场", "急场"],
            "火车站": ["火车占", "货车站"],
            "汽车站": ["汽车占", "气车站"],
            "公园": ["工园", "公元"],
            "广场": ["广厂", "管场"]
        }

        # 语气词和填充词纠正
        self.filler_corrections = {
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

    def load_config(self):
        """加载配置信息"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = {}
            logger.warning(f"配置文件 {self.config_file} 未找到，使用默认配置")
        except json.JSONDecodeError:
            self.config = {}
            logger.error(f"配置文件 {self.config_file} JSON格式错误，使用默认配置")

    def _clean_text(self, text: str) -> str:
        """基础文本清理"""
        text = text.strip()
        # 合并多个空格为一个
        text = re.sub(r'\s+', ' ', text)
        # 移除多余的换行符
        text = re.sub(r'\n+', ' ', text)
        return text

    def _correct_sound_alike_errors(self, text: str) -> str:
        """纠正音对字不对的错误"""
        for correct, wrong_list in self.sound_alike_errors.items():
            for wrong in wrong_list:
                text = text.replace(wrong, correct)
        
        # 电视剧专业词汇纠正
        for correct, wrong_list in self.tv_drama_terms.items():
            for wrong in wrong_list:
                text = text.replace(wrong, correct)
        
        return text

    def _correct_professional_terms(self, text: str) -> str:
        """纠正专业术语"""
        professional_terms = self.config.get("professional_terms", {})

        for correct_term, incorrect_terms in professional_terms.items():
            for incorrect_term in incorrect_terms:
                text = text.replace(incorrect_term, correct_term)
        return text

    def _correct_polyphones(self, text: str) -> str:
        """根据上下文纠正多音字"""
        polyphone_corrections = self.config.get("polyphone_corrections", {})

        for correct_word, context_info in polyphone_corrections.items():
            # 兼容新旧格式
            if "错误" in context_info:
                incorrect_words = context_info["错误"]
                keywords = context_info.get("关键词", [])
            else:
                incorrect_words = context_info.get("incorrect", [])
                keywords = context_info.get("keywords", [])

            # 如果上下文中存在关键词，则进行替换
            if any(keyword in text for keyword in keywords):
                for incorrect_word in incorrect_words:
                    text = text.replace(incorrect_word, correct_word)
        return text

    def _process_numbers_and_units(self, text: str) -> str:
        """处理数字和单位"""
        number_unit_corrections = self.config.get("number_unit_corrections", {})

        for correct_form, incorrect_forms in number_unit_corrections.items():
            for incorrect_form in incorrect_forms:
                text = text.replace(incorrect_form, correct_form)
        
        # 数字读音纠正
        digit_corrections = {
            "零": ["另", "令"],
            "一": ["医", "衣"],
            "二": ["尔", "而"],
            "三": ["伞", "散"],
            "四": ["是", "死"],
            "五": ["我", "吴"],
            "六": ["留", "流"],
            "七": ["期", "齐"],
            "八": ["吧", "把"],
            "九": ["就", "酒"],
            "十": ["是", "实"]
        }
        
        for correct, wrong_list in digit_corrections.items():
            for wrong in wrong_list:
                # 只在数字上下文中替换
                text = re.sub(f'({wrong})([十百千万])', f'{correct}\\2', text)
                text = re.sub(f'([十百千万])({wrong})', f'\\1{correct}', text)

        return text

    def _correct_filler_words(self, text: str) -> str:
        """纠正语气词和填充词"""
        for correct, wrong in self.filler_corrections.items():
            text = text.replace(wrong, correct)

        # 处理重复的语气词
        text = re.sub(r'([啊哦嗯额呃哈呵嘿])\1{2,}', r'\1', text)
        
        # 处理重复的字词（超过2次重复）
        text = re.sub(r'(.)\1{3,}', r'\1\1', text)

        return text

    def _smart_sentence_segmentation(self, text: str) -> str:
        """智能断句"""
        if not text:
            return text

        # 使用jieba分词
        words = list(jieba.cut(text))
        result = []

        # 定义各种语言标记词
        pause_words = ['那么', '然后', '接着', '于是', '所以', '因此', '另外', '还有', '而且', '并且', '同时', '此外', '总之', '最后', '最终']
        transition_words = ['但是', '不过', '然而', '可是', '只是', '只不过', '虽然', '尽管', '即使', '哪怕', '无论']
        question_words = ['什么', '哪里', '怎么', '为什么', '怎样', '如何', '多少', '几', '吗', '呢', '啊']
        exclamation_words = ['太', '真', '好', '很', '非常', '特别', '极其', '相当', '十分']
        
        sentence_enders = ['。', '！', '？']
        
        i = 0
        while i < len(words):
            word = words[i]
            result.append(word)

            # 在停顿词后添加逗号
            if word in pause_words and i < len(words) - 1:
                next_word = words[i + 1]
                if not next_word.startswith(('，', '。', '！', '？')):
                    result.append('，')

            # 在转折词前添加逗号（如果前面没有标点）
            if i > 0 and word in transition_words:
                if len(result) >= 2 and not result[-2].endswith(('，', '。', '！', '？')):
                    result.insert(-1, '，')

            # 处理疑问句
            if word in question_words:
                # 检查是否已经在句末
                remaining_words = words[i+1:]
                if not remaining_words or (len(remaining_words) <= 2 and all(w in ['吗', '呢', '啊', '的'] for w in remaining_words)):
                    if i == len(words) - 1 or (i == len(words) - 2 and words[i+1] in ['吗', '呢', '啊']):
                        result.append('？')
                        if i < len(words) - 1:
                            i += 1  # 跳过语气词

            # 处理感叹句
            if word in exclamation_words and i < len(words) - 1:
                # 检查后面是否有形容词或副词
                next_words = words[i+1:i+3]
                if any(w in ['好', '棒', '厉害', '了不起', '不错', '完美'] for w in next_words):
                    # 在句末添加感叹号
                    j = i + 1
                    while j < len(words) and words[j] not in sentence_enders:
                        j += 1
                    if j == len(words):  # 到了句末
                        result.append('！')

            i += 1

        text = ''.join(result)

        # 处理长句子，自动分段
        sentences = re.split(r'([。！？])', text)
        processed_sentences = []

        for sentence in sentences:
            if sentence and sentence not in ['。', '！', '？']:
                # 如果句子太长（超过30个字符）且没有标点，添加逗号分隔
                if len(sentence) > 30 and '，' not in sentence:
                    # 在连词处分隔
                    sentence = re.sub(r'(然后|接着|于是|所以|因此|但是|不过|然而)', r'，\1', sentence)
                    sentence = re.sub(r'(而且|并且|同时|另外|还有)', r'，\1', sentence)

                # 如果句子长度超过一定长度且没有标点，添加句号
                if len(sentence) > 15 and not sentence.endswith(('。', '！', '？', '，')):
                    sentence += '。'
            
            processed_sentences.append(sentence)

        return ''.join(processed_sentences)

    def _process_punctuation(self, text: str) -> str:
        """处理标点符号"""
        # 移除多余的空格
        text = re.sub(r'\s+', '', text)

        # 处理连续的标点符号
        text = re.sub(r'[，]{2,}', '，', text)
        text = re.sub(r'[。]{2,}', '。', text)
        text = re.sub(r'[！]{2,}', '！', text)
        text = re.sub(r'[？]{2,}', '？', text)

        # 处理标点符号前的空格
        text = re.sub(r'\s+([，。！？；：])', r'\1', text)

        # 修正标点符号的使用
        # 疑问句应该以问号结尾
        text = re.sub(r'([什么哪里怎么为什么怎样如何多少几])([^？]*?)([。！])', r'\1\2？', text)
        
        # 感叹句处理
        text = re.sub(r'([太真好很非常特别])([^！]*?)([。？])', r'\1\2！', text)

        # 确保句子有合适的结尾
        if text and not text.endswith(('。', '！', '？')):
            # 如果最后是逗号，改为句号
            if text.endswith('，'):
                text = text[:-1] + '。'
            else:
                # 根据语境决定标点
                if any(qw in text[-10:] for qw in ['什么', '怎么', '为什么', '吗', '呢']):
                    text += '？'
                elif any(ew in text[-10:] for ew in ['太', '真', '好', '很', '非常']):
                    text += '！'
                else:
                    text += '。'

        return text

    def post_process(self, text: str) -> str:
        """主要的文本后处理方法"""
        if not text or not text.strip():
            return text

        original_text = text
        
        # 1. 基础清理
        text = self._clean_text(text)

        # 2. 纠正音对字不对的错误
        text = self._correct_sound_alike_errors(text)

        # 3. 纠正语气词和填充词
        text = self._correct_filler_words(text)

        # 4. 专业术语纠正
        text = self._correct_professional_terms(text)

        # 5. 多音字纠正  
        text = self._correct_polyphones(text)

        # 6. 数字和单位处理
        text = self._process_numbers_and_units(text)

        # 7. 智能断句
        text = self._smart_sentence_segmentation(text)

        # 8. 最终标点符号处理
        text = self._process_punctuation(text)

        # 统计修改情况
        changes = self._count_changes(original_text, text)
        if changes > 0:
            logger.info(f"文本后处理完成，共修正 {changes} 处")

        return text.strip()

    def _count_changes(self, original: str, processed: str) -> int:
        """统计文本变化数量"""
        if len(original) == 0 and len(processed) == 0:
            return 0
        
        # 使用简单的字符差异计算
        changes = 0
        max_len = max(len(original), len(processed))
        min_len = min(len(original), len(processed))
        
        for i in range(min_len):
            if original[i] != processed[i]:
                changes += 1
        
        # 添加长度差异
        changes += abs(len(original) - len(processed))
        
        return changes

    def get_correction_stats(self, text: str) -> Dict[str, int]:
        """获取纠错统计信息"""
        stats = {
            "sound_alike_errors": 0,
            "professional_terms": 0,
            "polyphone_errors": 0,
            "number_units": 0,
            "filler_words": 0,
            "total_chars": len(text)
        }

        # 统计音对字不对错误
        for correct, wrong_list in self.sound_alike_errors.items():
            for wrong in wrong_list:
                stats["sound_alike_errors"] += text.count(wrong)

        # 统计专业名词错误
        professional_terms = self.config.get("professional_terms", {})
        for correct_term, wrong_terms in professional_terms.items():
            for wrong_term in wrong_terms:
                stats["professional_terms"] += text.count(wrong_term)

        # 统计多音字错误
        polyphone_corrections = self.config.get("polyphone_corrections", {})
        for correct_word, info in polyphone_corrections.items():
            wrong_words = info.get("incorrect", info.get("错误", []))
            for wrong_word in wrong_words:
                stats["polyphone_errors"] += text.count(wrong_word)

        # 统计数字单位错误
        number_unit_corrections = self.config.get("number_unit_corrections", {})
        for correct, wrong_list in number_unit_corrections.items():
            for wrong in wrong_list:
                stats["number_units"] += text.count(wrong)

        # 统计填充词
        for correct, wrong in self.filler_corrections.items():
            stats["filler_words"] += text.count(wrong)

        return stats

    def add_custom_correction(self, correct_word: str, wrong_words: List[str], category: str = "custom"):
        """添加自定义纠错词汇"""
        if category == "sound_alike":
            self.sound_alike_errors[correct_word] = wrong_words
        elif category == "tv_drama":
            self.tv_drama_terms[correct_word] = wrong_words
        elif category == "filler":
            for wrong in wrong_words:
                self.filler_corrections[wrong] = correct_word
        
        logger.info(f"添加自定义纠错词汇 ({category}): {correct_word} <- {wrong_words}")
