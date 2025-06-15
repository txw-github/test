
import re
import json
import os
import jieba
import zhon.hanzi
from typing import Dict, List, Tuple, Optional
import logging
import difflib

logger = logging.getLogger(__name__)

class AdvancedTextPostProcessor:
    """高级文本后处理器 - 多层次智能纠错系统"""

    def __init__(self, config_file: str = "text_correction_config.json"):
        self.config_file = config_file
        self.load_config()

        # 初始化结巴分词
        jieba.initialize()

        # 扩展的音对字不对错误词典
        self.sound_alike_errors = {
            # 日常用语
            "在哪里": ["在那里", "在纳里", "再哪里"],
            "怎么样": ["怎么羊", "怎么央", "怎莫样"],
            "什么时候": ["什么事后", "什么时后", "神马时候"],
            "为什么": ["为甚么", "围什么", "喂什么"],
            "这样子": ["这羊子", "这样紫", "这洋子"],
            "那样子": ["那羊子", "那样紫", "那洋子"],
            "没关系": ["没观系", "没关细", "没关西"],
            "不要紧": ["不要金", "不要斤", "不药紧"],
            "有意思": ["有意识", "有意丝", "又意思"],
            "没意思": ["没意识", "没意丝", "美意思"],
            "不好意思": ["不好意识", "不好意丝", "不号意思"],
            "一会儿": ["一会而", "一回儿", "一会尔"],
            "过一会儿": ["过一会而", "过一回儿", "过一会尔"],
            "等一下": ["等以下", "等一夏", "等以夏"],
            "等一等": ["等一灯", "等一邓", "等一登"],
            
            # 时间相关
            "现在": ["县在", "现再", "显在"],
            "以前": ["以钱", "已前", "移前"],
            "以后": ["以厚", "已后", "移后"],
            "刚才": ["刚材", "刚菜", "刚财"],
            "马上": ["马伤", "马商", "码上"],
            "立刻": ["立客", "立克", "力刻"],
            "突然": ["突燃", "突染", "凸然"],
            "忽然": ["忽燃", "忽染", "乎然"],
            
            # 情感表达
            "高兴": ["高性", "告兴", "搞兴"],
            "兴奋": ["性奋", "醒奋", "星奋"],
            "激动": ["即动", "急动", "击动"],
            "紧张": ["金张", "近张", "进张"],
            "着急": ["着记", "着及", "着急"],
            "担心": ["单心", "弹心", "淡心"],
            "放心": ["方心", "防心", "房心"],
            "开心": ["开新", "开信", "凯心"],
            "伤心": ["伤新", "商心", "上心"],
            "难过": ["难国", "男过", "难锅"],
            "痛苦": ["通苦", "痛哭", "同苦"],
            
            # 程度副词
            "特别": ["特白", "特北", "特贝"],
            "尤其": ["尤齐", "犹其", "由其"],
            "非常": ["飞常", "非长", "费常"],
            "很": ["狠", "恨", "痕"],
            "太": ["泰", "态", "台"],
            "真": ["珍", "针", "震"],
            "挺": ["庭", "停", "听"],
            "相当": ["想当", "响当", "相党"],
            
            # 动作动词
            "看见": ["看间", "看建", "康见"],
            "听见": ["听间", "听建", "停见"],
            "遇见": ["遇间", "遇建", "预见"],
            "发现": ["发县", "发献", "法现"],
            "知道": ["治道", "之道", "智道"],
            "明白": ["明百", "名白", "明拜"],
            "清楚": ["清初", "清出", "情楚"],
            "记得": ["记的", "记德", "计得"],
            "忘记": ["忘计", "望记", "王记"],
            "想起": ["想齐", "向起", "想气"],
            
            # 关系词汇
            "因为": ["音为", "因围", "因维"],
            "所以": ["所已", "索以", "锁以"],
            "如果": ["如国", "入果", "汝果"],
            "虽然": ["虽燃", "随然", "岁然"],
            "但是": ["但事", "蛋是", "旦是"],
            "不过": ["不锅", "不国", "不郭"],
            "然而": ["然二", "燃而", "染而"],
            "而且": ["而切", "尔且", "儿且"],
            "并且": ["并切", "病且", "兵且"],
            "况且": ["况切", "狂且", "矿且"],
            
            # 方位词
            "上面": ["上边", "商面", "伤面"],
            "下面": ["下边", "夏面", "下棉"],
            "左边": ["做边", "左变", "佐边"],
            "右边": ["有边", "右变", "又边"],
            "前面": ["前边", "钱面", "前棉"],
            "后面": ["后边", "厚面", "候面"],
            "里面": ["里边", "理面", "礼面"],
            "外面": ["外边", "外棉", "歪面"],
            "中间": ["中键", "钟间", "中尖"],
            "旁边": ["旁变", "胖边", "螃边"],
            
            # 人称代词
            "我们": ["我门", "沃们", "握们"],
            "你们": ["你门", "尼们", "泥们"],
            "他们": ["他门", "它们", "她们"],
            "自己": ["字己", "自记", "紫己"],
            "别人": ["别仁", "别任", "贝人"],
            "大家": ["大嘉", "达家", "大价"],
            
            # 疑问词
            "什么": ["神马", "甚么", "什莫"],
            "怎么": ["咋么", "怎莫", "咋莫"],
            "为什么": ["为神马", "为甚么", "为什莫"],
            "哪里": ["那里", "纳里", "那力"],
            "哪儿": ["那儿", "纳儿", "那尔"],
            "怎样": ["咋样", "怎洋", "咋洋"],
            "如何": ["汝何", "入何", "儒何"],
            "多少": ["多绍", "多烧", "朵少"],
            
            # 数量词
            "一些": ["一写", "一谢", "衣些"],
            "几个": ["级个", "几歌", "记个"],
            "很多": ["狠多", "痕多", "恨多"],
            "不少": ["不绍", "不烧", "布少"],
            "许多": ["续多", "徐多", "需多"],
            "大量": ["大凉", "大两", "达量"],
            
            # 情况状态
            "情况": ["请况", "情框", "清况"],
            "状况": ["壮况", "状框", "装况"],
            "问题": ["文题", "问踢", "稳题"],
            "办法": ["办发", "版法", "办罚"],
            "方法": ["方发", "房法", "芳法"],
            "方式": ["方是", "芳式", "方士"],
            "方面": ["方边", "芳面", "方棉"],
            "结果": ["解果", "结国", "街果"],
            "原因": ["元因", "原音", "圆因"],
            "原来": ["元来", "原莱", "缘来"],
            
            # 表达词汇
            "表示": ["表是", "表适", "表事"],
            "表现": ["表县", "表显", "表献"],
            "说明": ["说名", "说铭", "硕明"],
            "解释": ["解失", "街释", "解湿"],
            "证明": ["正明", "症明", "挣明"],
            "肯定": ["肯丁", "肯订", "恳定"],
            "否定": ["否丁", "否订", "佛定"],
            "确定": ["确丁", "确订", "雀定"],
            "决定": ["绝定", "觉定", "决订"],
            
            # 家庭关系
            "父母": ["父穆", "夫母", "父木"],
            "爸爸": ["八八", "巴巴", "拔拔"],
            "妈妈": ["马马", "骂骂", "麻麻"],
            "爷爷": ["夜夜", "也也", "野野"],
            "奶奶": ["乃乃", "奈奈", "耐耐"],
            "哥哥": ["歌歌", "葛葛", "个个"],
            "姐姐": ["街街", "接接", "节节"],
            "弟弟": ["第第", "地地", "低低"],
            "妹妹": ["美美", "没没", "媚媚"],
            "叔叔": ["书书", "舒舒", "属属"],
            "阿姨": ["啊姨", "阿一", "阿衣"],
            
            # 职业相关
            "老师": ["老是", "老师", "劳师"],
            "学生": ["雪生", "学声", "学升"],
            "医生": ["一生", "衣生", "易生"],
            "护士": ["户士", "护是", "户是"],
            "警察": ["井察", "警擦", "京察"],
            "司机": ["丝机", "思机", "四机"],
            "工人": ["公人", "工任", "攻人"],
            "农民": ["浓民", "弄民", "农明"],
            "商人": ["伤人", "商任", "上人"],
            "老板": ["老版", "老伴", "老半"],
        }

        # 电视剧专业词汇扩展
        self.tv_drama_terms = {
            # 演艺圈
            "演员": ["言员", "眼员", "烟员", "演园"],
            "导演": ["道演", "倒演", "到演", "导言"],
            "编剧": ["编据", "编局", "变剧", "编具"],
            "制片人": ["制片仁", "制篇人", "制片任"],
            "制片": ["制篇", "制片", "治片"],
            "监制": ["监治", "简制", "监智"],
            "出品人": ["出品仁", "出品任", "出瓶人"],
            "发行": ["发型", "发行", "法行"],
            "投资": ["头资", "投子", "头子"],
            "票房": ["票方", "票防", "漂房"],
            
            # 剧本创作
            "剧本": ["据本", "剧奔", "局本", "剧笨"],
            "剧情": ["据情", "举情", "局情", "剧清"],
            "情节": ["情结", "清洁", "情街", "情节"],
            "故事": ["古事", "股事", "故是", "顾事"],
            "台词": ["台词", "抬词", "台慈", "太词"],
            "对白": ["对百", "对摆", "队白", "对拜"],
            "独白": ["独百", "独摆", "读白", "毒白"],
            "旁白": ["旁百", "胖白", "螃白", "旁拜"],
            "字幕": ["字慕", "字墓", "自幕", "字母"],
            "配音": ["陪音", "配因", "佩音", "培音"],
            
            # 角色相关
            "主角": ["主脚", "朱角", "猪角", "住角"],
            "配角": ["陪角", "配脚", "佩角", "培角"],
            "反派": ["反派", "反牌", "翻派", "返派"],
            "正面": ["正棉", "证面", "争面", "整面"],
            "反面": ["反棉", "返面", "翻面", "饭面"],
            "角色": ["脚色", "叫色", "觉色", "角瑟"],
            "人物": ["人勿", "任物", "人物", "仁物"],
            "性格": ["性各", "星格", "姓格", "醒格"],
            "特点": ["特店", "特点", "特典", "特电"],
            "背景": ["被景", "倍景", "背井", "北景"],
            
            # 拍摄制作
            "拍摄": ["拍设", "派摄", "排摄", "拍射"],
            "摄影": ["设影", "射影", "摄引", "摄营"],
            "录制": ["路制", "露制", "绿制", "陆制"],
            "剪辑": ["简辑", "减辑", "见辑", "剪集"],
            "后期": ["后期", "厚期", "候期", "后旗"],
            "特效": ["特校", "特笑", "特孝", "特效"],
            "配乐": ["陪乐", "培乐", "佩乐", "配月"],
            "音效": ["音校", "音笑", "阴效", "音孝"],
            "灯光": ["等光", "登光", "灯关", "灯广"],
            "化妆": ["化装", "画妆", "化状", "化庄"],
            
            # 场景道具
            "场景": ["厂景", "场井", "常景", "长景"],
            "布景": ["部景", "布井", "不景", "步景"],
            "道具": ["到具", "道句", "倒具", "导具"],
            "服装": ["服状", "服庄", "伏装", "福装"],
            "造型": ["造形", "早型", "燥型", "躁型"],
            "妆容": ["装容", "状容", "庄容", "妆融"],
            "发型": ["发形", "法型", "罚型", "乏型"],
            "饰品": ["事品", "食品", "试品", "适品"],
            "首饰": ["手饰", "守饰", "受饰", "收饰"],
            "珠宝": ["猪宝", "朱宝", "竹宝", "株宝"],
            
            # 播出发行
            "播出": ["报出", "播初", "波出", "拨出"],
            "首播": ["手播", "守播", "受播", "收播"],
            "重播": ["种播", "中播", "虫播", "冲播"],
            "点播": ["店播", "电播", "典播", "滴播"],
            "收视": ["收事", "收是", "守视", "受视"],
            "收视率": ["收事率", "收是率", "守视率"],
            "观众": ["观重", "关众", "管众", "观仲"],
            "观看": ["观刊", "关看", "管看", "观砍"],
            "评价": ["平价", "评家", "瓶价", "评嘉"],
            "评论": ["平论", "评轮", "瓶论", "评伦"],
            "口碑": ["口杯", "口被", "口悲", "口备"]
        }

        # 同音字错误扩展
        self.homophone_errors = {
            # 的得地
            "的": ["地", "得"],
            "地": ["的", "得"],
            "得": ["的", "地"],
            
            # 在再
            "在": ["再", "栽"],
            "再": ["在", "载"],
            
            # 做作
            "做": ["作", "坐"],
            "作": ["做", "座"],
            
            # 进近
            "进": ["近", "金"],
            "近": ["进", "紧"],
            
            # 只支知
            "只": ["支", "知"],
            "支": ["只", "知"],
            "知": ["只", "支"],
            
            # 像向象
            "像": ["向", "象"],
            "向": ["像", "象"],
            "象": ["像", "向"],
            
            # 两量
            "两": ["量", "亮"],
            "量": ["两", "亮"],
            
            # 和合
            "和": ["合", "河"],
            "合": ["和", "何"],
            
            # 已以
            "已": ["以", "意"],
            "以": ["已", "意"],
            
            # 处出
            "处": ["出", "初"],
            "出": ["处", "初"],
            
            # 与于
            "与": ["于", "鱼"],
            "于": ["与", "鱼"]
        }

        # 形近字错误
        self.similar_shape_errors = {
            "密": ["蜜", "秘"],
            "蜜": ["密", "秘"],
            "秘": ["密", "蜜"],
            
            "账": ["帐", "账"],
            "帐": ["账", "帐"],
            
            "采": ["彩", "睬"],
            "彩": ["采", "睬"],
            "睬": ["采", "彩"],
            
            "辨": ["辩", "瓣"],
            "辩": ["辨", "瓣"],
            "瓣": ["辨", "辩"],
            
            "象": ["像", "橡"],
            "像": ["象", "橡"],
            "橡": ["象", "像"],
            
            "暴": ["爆", "瀑"],
            "爆": ["暴", "瀑"],
            "瀑": ["暴", "爆"]
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
            "行行行": "行",
            "哎呀呀": "哎呀",
            "啦啦啦": "啦",
            "哦哦哦": "哦",
            "唉唉唉": "唉"
        }

        # 上下文语义纠错规则
        self.context_rules = {
            # 情感类
            "高兴": {
                "keywords": ["开心", "快乐", "激动", "兴奋", "喜悦"],
                "wrong_words": ["高性", "告兴", "搞兴"]
            },
            "伤心": {
                "keywords": ["难过", "痛苦", "哭", "眼泪", "悲伤"],
                "wrong_words": ["伤新", "商心", "上心"]
            },
            
            # 时间类
            "以前": {
                "keywords": ["过去", "从前", "曾经", "以往"],
                "wrong_words": ["以钱", "已前", "移前"]
            },
            "以后": {
                "keywords": ["将来", "未来", "今后"],
                "wrong_words": ["以厚", "已后", "移后"]
            },
            
            # 地点类
            "医院": {
                "keywords": ["医生", "护士", "病人", "治疗", "手术", "住院"],
                "wrong_words": ["医源", "一院", "衣院"]
            },
            "学校": {
                "keywords": ["老师", "学生", "上课", "教室", "学习"],
                "wrong_words": ["学效", "雪校", "学孝"]
            }
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

    def preprocess_text(self, text: str) -> str:
        """预处理文本 - 第一阶段处理"""
        if not text or not text.strip():
            return text

        # 1. 移除异常字符
        text = re.sub(r'[^\u4e00-\u9fff\u3400-\u4dbf\w\s\.\,\!\?\:\;\-\(\)\[\]\"\'。，！？：；（）【】""''、]', '', text)
        
        # 2. 标准化空白字符
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        
        # 3. 修正明显的重复字符
        text = re.sub(r'(.)\1{3,}', r'\1', text)  # 超过3个重复的字符减少为1个
        
        # 4. 处理常见的语音识别错误
        text = re.sub(r'[啊]{2,}', '啊', text)
        text = re.sub(r'[嗯]{2,}', '嗯', text)
        text = re.sub(r'[哦]{2,}', '哦', text)
        
        return text.strip()

    def correct_sound_alike_errors(self, text: str) -> str:
        """纠正音对字不对的错误"""
        corrections_made = 0
        
        # 使用扩展的音对字不对词典
        for correct, wrong_list in self.sound_alike_errors.items():
            for wrong in wrong_list:
                if wrong in text:
                    text = text.replace(wrong, correct)
                    corrections_made += 1
        
        # 电视剧专业词汇纠正
        for correct, wrong_list in self.tv_drama_terms.items():
            for wrong in wrong_list:
                if wrong in text:
                    text = text.replace(wrong, correct)
                    corrections_made += 1
        
        if corrections_made > 0:
            logger.debug(f"音对字不对纠错: {corrections_made} 处")
        
        return text

    def correct_homophone_errors(self, text: str) -> str:
        """纠正同音字错误"""
        corrections_made = 0
        
        # 使用上下文判断同音字
        words = jieba.lcut(text)
        corrected_words = []
        
        for i, word in enumerate(words):
            corrected_word = word
            
            # 检查是否是同音字错误
            for correct_char, wrong_chars in self.homophone_errors.items():
                if word in wrong_chars:
                    # 分析上下文
                    context = " ".join(words[max(0, i-2):i+3])
                    
                    # 简单的上下文匹配
                    if self._should_use_char(correct_char, context):
                        corrected_word = correct_char
                        corrections_made += 1
                        break
            
            corrected_words.append(corrected_word)
        
        if corrections_made > 0:
            logger.debug(f"同音字纠错: {corrections_made} 处")
        
        return "".join(corrected_words)

    def _should_use_char(self, char: str, context: str) -> bool:
        """判断在特定上下文中是否应该使用某个字符"""
        # 这里可以实现更复杂的语义判断
        # 现在使用简单的关键词匹配
        
        char_contexts = {
            "的": ["我的", "他的", "她的", "它的", "你的", "我们的"],
            "地": ["慢慢地", "快快地", "静静地", "悄悄地"],
            "得": ["跑得快", "说得好", "做得对", "走得慢"],
            "在": ["在家", "在学校", "在公司", "正在"],
            "再": ["再见", "再次", "再来", "再说"],
            "做": ["做饭", "做事", "做工作"],
            "作": ["工作", "作业", "作文", "创作"]
        }
        
        if char in char_contexts:
            return any(keyword in context for keyword in char_contexts[char])
        
        return True

    def correct_similar_shape_errors(self, text: str) -> str:
        """纠正形近字错误"""
        corrections_made = 0
        
        for correct, wrong_list in self.similar_shape_errors.items():
            for wrong in wrong_list:
                if wrong in text:
                    # 这里可以添加更智能的判断逻辑
                    text = text.replace(wrong, correct)
                    corrections_made += 1
        
        if corrections_made > 0:
            logger.debug(f"形近字纠错: {corrections_made} 处")
        
        return text

    def context_semantic_correction(self, text: str) -> str:
        """基于上下文的语义纠错"""
        corrections_made = 0
        
        for correct_word, rule in self.context_rules.items():
            keywords = rule.get("keywords", [])
            wrong_words = rule.get("wrong_words", [])
            
            # 检查上下文中是否包含相关关键词
            context_match = any(keyword in text for keyword in keywords)
            
            if context_match:
                for wrong_word in wrong_words:
                    if wrong_word in text:
                        text = text.replace(wrong_word, correct_word)
                        corrections_made += 1
        
        if corrections_made > 0:
            logger.debug(f"语义纠错: {corrections_made} 处")
        
        return text

    def correct_professional_terms(self, text: str) -> str:
        """纠正专业术语"""
        corrections_made = 0
        professional_terms = self.config.get("professional_terms", {})

        for correct_term, incorrect_terms in professional_terms.items():
            for incorrect_term in incorrect_terms:
                if incorrect_term in text:
                    text = text.replace(incorrect_term, correct_term)
                    corrections_made += 1
        
        if corrections_made > 0:
            logger.debug(f"专业术语纠错: {corrections_made} 处")
        
        return text

    def correct_polyphones(self, text: str) -> str:
        """根据上下文纠正多音字"""
        corrections_made = 0
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
                    if incorrect_word in text:
                        text = text.replace(incorrect_word, correct_word)
                        corrections_made += 1
        
        if corrections_made > 0:
            logger.debug(f"多音字纠错: {corrections_made} 处")
        
        return text

    def process_numbers_and_units(self, text: str) -> str:
        """处理数字和单位"""
        corrections_made = 0
        number_unit_corrections = self.config.get("number_unit_corrections", {})

        for correct_form, incorrect_forms in number_unit_corrections.items():
            for incorrect_form in incorrect_forms:
                if incorrect_form in text:
                    text = text.replace(incorrect_form, correct_form)
                    corrections_made += 1
        
        # 扩展数字读音纠正
        digit_corrections = {
            "零": ["另", "令", "灵"],
            "一": ["医", "衣", "义"],
            "二": ["尔", "而", "儿"],
            "三": ["伞", "散", "山"],
            "四": ["是", "死", "思"],
            "五": ["我", "吴", "武"],
            "六": ["留", "流", "柳"],
            "七": ["期", "齐", "起"],
            "八": ["吧", "把", "巴"],
            "九": ["就", "酒", "久"],
            "十": ["是", "实", "石"]
        }
        
        for correct, wrong_list in digit_corrections.items():
            for wrong in wrong_list:
                # 只在数字上下文中替换
                pattern = f'({wrong})([十百千万亿])'
                text = re.sub(pattern, f'{correct}\\2', text)
                pattern = f'([十百千万亿])({wrong})'
                text = re.sub(pattern, f'\\1{correct}', text)
                
                if pattern in text:
                    corrections_made += 1

        if corrections_made > 0:
            logger.debug(f"数字单位纠错: {corrections_made} 处")

        return text

    def correct_filler_words(self, text: str) -> str:
        """纠正语气词和填充词"""
        corrections_made = 0
        
        for correct, wrong in self.filler_corrections.items():
            if wrong in text:
                text = text.replace(wrong, correct)
                corrections_made += 1

        # 处理重复的语气词
        original_text = text
        text = re.sub(r'([啊哦嗯额呃哈呵嘿唉])\1{2,}', r'\1', text)
        
        # 处理重复的字词（超过2次重复）
        text = re.sub(r'(.)\1{3,}', r'\1\1', text)
        
        if text != original_text:
            corrections_made += 1

        if corrections_made > 0:
            logger.debug(f"语气词填充词纠错: {corrections_made} 处")

        return text

    def smart_sentence_segmentation(self, text: str) -> str:
        """智能断句 - 增强版"""
        if not text:
            return text

        # 使用jieba分词
        words = list(jieba.cut(text))
        result = []

        # 扩展各种语言标记词
        pause_words = ['那么', '然后', '接着', '于是', '所以', '因此', '另外', '还有', '而且', '并且', '同时', '此外', '总之', '最后', '最终', '首先', '其次', '再次', '最重要的是']
        transition_words = ['但是', '不过', '然而', '可是', '只是', '只不过', '虽然', '尽管', '即使', '哪怕', '无论', '不管', '除非', '否则']
        question_words = ['什么', '哪里', '怎么', '为什么', '怎样', '如何', '多少', '几', '吗', '呢', '啊', '哪', '谁', '何时', '何地']
        exclamation_words = ['太', '真', '好', '很', '非常', '特别', '极其', '相当', '十分', '超级', '巨', '超']
        
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
                if not remaining_words or (len(remaining_words) <= 2 and all(w in ['吗', '呢', '啊', '的', '了'] for w in remaining_words)):
                    if i == len(words) - 1 or (i == len(words) - 2 and words[i+1] in ['吗', '呢', '啊']):
                        result.append('？')
                        if i < len(words) - 1:
                            i += 1  # 跳过语气词

            # 处理感叹句
            if word in exclamation_words and i < len(words) - 1:
                # 检查后面是否有形容词或副词
                next_words = words[i+1:i+4]
                if any(w in ['好', '棒', '厉害', '了不起', '不错', '完美', '棒极了', '太好了'] for w in next_words):
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
                # 如果句子太长（超过40个字符）且没有标点，添加逗号分隔
                if len(sentence) > 40 and '，' not in sentence:
                    # 在连词处分隔
                    sentence = re.sub(r'(然后|接着|于是|所以|因此|但是|不过|然而)', r'，\1', sentence)
                    sentence = re.sub(r'(而且|并且|同时|另外|还有)', r'，\1', sentence)

                # 如果句子长度超过一定长度且没有标点，添加句号
                if len(sentence) > 20 and not sentence.endswith(('。', '！', '？', '，')):
                    sentence += '。'
            
            processed_sentences.append(sentence)

        return ''.join(processed_sentences)

    def process_punctuation(self, text: str) -> str:
        """处理标点符号 - 增强版"""
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
        text = re.sub(r'([什么哪里怎么为什么怎样如何多少几谁])([^？]*?)([。！])', r'\1\2？', text)
        
        # 感叹句处理
        text = re.sub(r'([太真好很非常特别])([^！]*?)([。？])', r'\1\2！', text)

        # 处理对话中的冒号
        text = re.sub(r'([说道讲告诉回答])([^：]*?)([。！？])', r'\1：\2', text)

        # 确保句子有合适的结尾
        if text and not text.endswith(('。', '！', '？')):
            # 如果最后是逗号，改为句号
            if text.endswith('，'):
                text = text[:-1] + '。'
            else:
                # 根据语境决定标点
                if any(qw in text[-15:] for qw in ['什么', '怎么', '为什么', '吗', '呢', '哪里', '谁']):
                    text += '？'
                elif any(ew in text[-15:] for ew in ['太', '真', '好', '很', '非常', '超级', '巨']):
                    text += '！'
                else:
                    text += '。'

        return text

    def post_process(self, text: str) -> str:
        """主要的文本后处理方法 - 多层次处理"""
        if not text or not text.strip():
            return text

        original_text = text
        total_corrections = 0
        
        logger.debug(f"开始多层次文本后处理，原文长度: {len(text)}")

        # 第1层：预处理
        text = self.preprocess_text(text)

        # 第2层：音对字不对纠错
        text = self.correct_sound_alike_errors(text)

        # 第3层：同音字纠错
        text = self.correct_homophone_errors(text)

        # 第4层：形近字纠错
        text = self.correct_similar_shape_errors(text)

        # 第5层：上下文语义纠错
        text = self.context_semantic_correction(text)

        # 第6层：语气词和填充词纠错
        text = self.correct_filler_words(text)

        # 第7层：专业术语纠正
        text = self.correct_professional_terms(text)

        # 第8层：多音字纠正  
        text = self.correct_polyphones(text)

        # 第9层：数字和单位处理
        text = self.process_numbers_and_units(text)

        # 第10层：智能断句
        text = self.smart_sentence_segmentation(text)

        # 第11层：最终标点符号处理
        text = self.process_punctuation(text)

        # 统计修改情况
        changes = self._count_changes(original_text, text)
        if changes > 0:
            logger.info(f"多层次文本后处理完成，共修正 {changes} 处错误")

        return text.strip()

    def _count_changes(self, original: str, processed: str) -> int:
        """统计文本变化数量 - 使用编辑距离算法"""
        if len(original) == 0 and len(processed) == 0:
            return 0
        
        # 使用difflib计算更精确的差异
        matcher = difflib.SequenceMatcher(None, original, processed)
        changes = 0
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag in ['replace', 'delete', 'insert']:
                changes += max(i2 - i1, j2 - j1)
        
        return changes

    def get_correction_stats(self, text: str) -> Dict[str, int]:
        """获取纠错统计信息 - 增强版"""
        stats = {
            "sound_alike_errors": 0,
            "homophone_errors": 0,
            "similar_shape_errors": 0,
            "professional_terms": 0,
            "polyphone_errors": 0,
            "number_units": 0,
            "filler_words": 0,
            "context_errors": 0,
            "total_chars": len(text)
        }

        # 统计各类错误
        for correct, wrong_list in self.sound_alike_errors.items():
            for wrong in wrong_list:
                stats["sound_alike_errors"] += text.count(wrong)

        for correct, wrong_list in self.homophone_errors.items():
            for wrong in wrong_list:
                stats["homophone_errors"] += text.count(wrong)

        for correct, wrong_list in self.similar_shape_errors.items():
            for wrong in wrong_list:
                stats["similar_shape_errors"] += text.count(wrong)

        # 专业名词错误
        professional_terms = self.config.get("professional_terms", {})
        for correct_term, wrong_terms in professional_terms.items():
            for wrong_term in wrong_terms:
                stats["professional_terms"] += text.count(wrong_term)

        # 多音字错误
        polyphone_corrections = self.config.get("polyphone_corrections", {})
        for correct_word, info in polyphone_corrections.items():
            wrong_words = info.get("incorrect", info.get("错误", []))
            for wrong_word in wrong_words:
                stats["polyphone_errors"] += text.count(wrong_word)

        # 数字单位错误
        number_unit_corrections = self.config.get("number_unit_corrections", {})
        for correct, wrong_list in number_unit_corrections.items():
            for wrong in wrong_list:
                stats["number_units"] += text.count(wrong)

        # 填充词
        for correct, wrong in self.filler_corrections.items():
            stats["filler_words"] += text.count(wrong)

        # 上下文错误
        for correct_word, rule in self.context_rules.items():
            wrong_words = rule.get("wrong_words", [])
            for wrong_word in wrong_words:
                stats["context_errors"] += text.count(wrong_word)

        return stats

    def add_custom_correction(self, correct_word: str, wrong_words: List[str], category: str = "custom"):
        """添加自定义纠错词汇"""
        if category == "sound_alike":
            self.sound_alike_errors[correct_word] = wrong_words
        elif category == "homophone":
            self.homophone_errors[correct_word] = wrong_words
        elif category == "similar_shape":
            self.similar_shape_errors[correct_word] = wrong_words
        elif category == "tv_drama":
            self.tv_drama_terms[correct_word] = wrong_words
        elif category == "filler":
            for wrong in wrong_words:
                self.filler_corrections[wrong] = correct_word
        
        logger.info(f"添加自定义纠错词汇 ({category}): {correct_word} <- {wrong_words}")

# 为了保持向后兼容性，创建别名
TextPostProcessor = AdvancedTextPostProcessor
