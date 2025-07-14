
import re
import json
import os
import jieba
import zhon.hanzi
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class TextPostProcessor:
    """高级文本后处理器 - 专门优化中文识别结果"""
    
    def __init__(self):
        self.load_correction_data()
        
    def load_correction_data(self):
        """加载纠错数据"""

class TextPostProcessor:
    """增强版文本后处理器 - 专业处理中文语音识别错误"""

    def __init__(self, config_file: str = "text_correction_config.json"):
        self.config_file = config_file
        self.load_config()

        # 初始化结巴分词
        jieba.initialize()

        # 大幅扩展的同音字错误词典
        self.sound_alike_errors = {
            # 日常用语类
            "在哪里": ["在那里", "在纳里", "在哪理", "在那理"],
            "怎么样": ["怎么羊", "怎么央", "怎么养", "怎莫样"],
            "什么时候": ["什么事后", "什么时后", "什么时侯", "甚么时候"],
            "为什么": ["为甚么", "围什么", "为神马", "维什么"],
            "这样子": ["这羊子", "这样紫", "这洋子", "者样子"],
            "那样子": ["那羊子", "那样紫", "那洋子", "拿样子"],
            "没关系": ["没观系", "没关细", "没干系", "美关系"],
            "不要紧": ["不要金", "不要斤", "不要今", "布要紧"],
            "有意思": ["有意识", "有意丝", "有意事", "友意思"],
            "没意思": ["没意识", "没意丝", "没意事", "美意思"],
            "不好意思": ["不好意识", "不好意丝", "不好意事", "布好意思"],
            
            # 时间表达类
            "一会儿": ["一会而", "一回儿", "一会尔", "移会儿"],
            "过一会儿": ["过一会而", "过一回儿", "过移会儿", "国一会儿"],
            "等一下": ["等以下", "等一夏", "等一吓", "等益下"],
            "等一等": ["等一灯", "等一邓", "等一登", "等移等"],
            "现在": ["县在", "现再", "现栽", "显在"],
            "以前": ["以钱", "已前", "移前", "议前"],
            "以后": ["以厚", "已后", "移后", "议后"],
            "当时": ["当实", "当是", "当世", "党时"],
            
            # 礼貌用语类
            "再见": ["在见", "再间", "再建", "栽见"],
            "拜拜": ["白白", "败败", "摆摆", "百拜"],
            "谢谢": ["些些", "写写", "泄泄", "协协"],
            "对不起": ["对不齐", "对布起", "堆不起", "兑不起"],
            "麻烦": ["马烦", "妈烦", "码烦", "嘛烦"],
            "辛苦": ["心苦", "新苦", "辛库", "信苦"],
            
            # 情绪表达类
            "厉害": ["利害", "历害", "力害", "励害"],
            "表现": ["表县", "表显", "表线", "表先"],
            "表示": ["表是", "表适", "表事", "表式"],
            "注意": ["主意", "住意", "注益", "主义"],
            "小心": ["小新", "小信", "小辛", "校心"],
            "当心": ["当新", "当信", "当辛", "党心"],
            "担心": ["单心", "弹心", "胆心", "淡心"],
            "放心": ["方心", "防心", "房心", "芳心"],
            "开心": ["开新", "开信", "开辛", "凯心"],
            "高兴": ["高性", "告兴", "高星", "搞兴"],
            "兴奋": ["性奋", "醒奋", "星奋", "行奋"],
            "激动": ["即动", "急动", "极动", "击动"],
            "紧张": ["金张", "近张", "进张", "禁张"],
            "着急": ["着记", "着及", "着集", "者急"],
            
            # 程度副词类
            "特别": ["特白", "特北", "特贝", "特备"],
            "尤其": ["尤齐", "犹其", "优其", "由其"],
            "尤其是": ["尤齐是", "犹其是", "优其是", "由其是"],
            "关键": ["观键", "关剑", "关见", "关坚"],
            "重要": ["种要", "中要", "重药", "众要"],
            "主要": ["主药", "珠要", "住要", "朱要"],
            "首要": ["手要", "收要", "守要", "受要"],
            "必要": ["比要", "必药", "毕要", "避要"],
            "需要": ["须要", "需药", "徐要", "虚要"],
            
            # 确定性表达类
            "一定": ["一顶", "一订", "移定", "义定"],
            "肯定": ["肯丁", "肯订", "肯定", "垦定"],
            "当然": ["当染", "党然", "当燃", "档然"],
            "自然": ["自燃", "紫然", "资然", "子然"],
            "确实": ["确是", "确世", "确事", "确适"],
            "的确": ["的确", "得确", "地确", "滴确"],
            "真的": ["真地", "真得", "针的", "珍的"],
            "确定": ["确丁", "确订", "确定", "雀定"],
            
            # 方位时间类
            "以上": ["以商", "已上", "移上", "议上"],
            "以下": ["以夏", "已下", "移下", "议下"],
            "上面": ["上边", "上面", "商面", "伤面"],
            "下面": ["下边", "下面", "夏面", "吓面"],
            "前面": ["钱面", "前边", "前面", "钱边"],
            "后面": ["厚面", "后边", "后面", "候面"],
            "左边": ["作边", "左变", "坐边", "左编"],
            "右边": ["有边", "又边", "右变", "佑边"],
            "中间": ["中见", "中坚", "中监", "众间"],
            "附近": ["付近", "副近", "富近", "府近"],
            "旁边": ["胖边", "旁变", "膀边", "螃边"],
            
            # 数量表达类
            "许多": ["许朵", "需多", "虚多", "徐多"],
            "很多": ["恨多", "狠多", "痕多", "很朵"],
            "不少": ["不绍", "布少", "不小", "不邵"],
            "一些": ["一谢", "移些", "一协", "一写"],
            "有些": ["有谢", "友些", "有协", "有写"],
            "这些": ["这谢", "者些", "这协", "这写"],
            "那些": ["那谢", "拿些", "那协", "那写"],
            "几个": ["几各", "机个", "记个", "集个"],
            "每个": ["美个", "没个", "梅个", "每各"],
            "所有": ["所友", "锁有", "索有", "梭有"],
            
            # 动作表达类
            "开始": ["开事", "开是", "开适", "凯始"],
            "结束": ["结速", "结树", "结书", "界束"],
            "继续": ["即续", "基续", "及续", "机续"],
            "停止": ["停之", "停只", "停知", "听止"],
            "进行": ["近行", "进兴", "进星", "禁行"],
            "发生": ["发声", "发升", "法生", "发胜"],
            "出现": ["出县", "出显", "出线", "出先"],
            "发现": ["发县", "发显", "发线", "法现"],
            "发展": ["发占", "发站", "法展", "发战"],
            "完成": ["完城", "完诚", "完呈", "万成"],
            "实现": ["实县", "实显", "实线", "实先"],
            "达到": ["达道", "答到", "打到", "大到"],
            
            # 感官动词类
            "看见": ["看间", "看建", "看监", "刊见"],
            "听见": ["听间", "听建", "听监", "厅见"],
            "感觉": ["敢觉", "感角", "感脚", "干觉"],
            "觉得": ["角得", "脚得", "觉的", "觉地"],
            "认为": ["人为", "认围", "人围", "任为"],
            "以为": ["已为", "移为", "议为", "易为"],
            "知道": ["知到", "至道", "志道", "制道"],
            "明白": ["明百", "明摆", "名白", "鸣白"],
            "了解": ["了姐", "了街", "了结", "料解"],
            "理解": ["理姐", "理街", "里解", "力解"],
            
            # 思维动词类
            "想要": ["想药", "想摇", "像要", "响要"],
            "希望": ["西望", "稀望", "息望", "吸望"],
            "打算": ["打蒜", "大算", "打散", "答算"],
            "计划": ["机划", "即划", "计花", "记划"],
            "决定": ["绝定", "决丁", "决订", "觉定"],
            "选择": ["选者", "宣择", "选责", "选泽"],
            "考虑": ["靠虑", "考虚", "考绿", "拷虑"],
            "思考": ["私考", "四考", "死考", "丝考"],
            "记得": ["记的", "记地", "及得", "机得"],
            "忘记": ["忘及", "忘记", "王记", "网记"],
            
            # 状态形容词类
            "舒服": ["书服", "属服", "疏服", "熟服"],
            "难受": ["南受", "男受", "难守", "难授"],
            "容易": ["融易", "荣易", "溶易", "蓉易"],
            "困难": ["困南", "困男", "捆难", "昆难"],
            "简单": ["间单", "建单", "监单", "检单"],
            "复杂": ["复在", "复栽", "复载", "富杂"],
            "清楚": ["清处", "清除", "青楚", "情楚"],
            "模糊": ["模乎", "摸糊", "模胡", "魔糊"],
            "明显": ["明县", "名显", "鸣显", "明线"],
            "清晰": ["清西", "清系", "清息", "青晰"],
            
            # 评价形容词类
            "满意": ["满义", "满益", "满议", "慢意"],
            "失望": ["失忘", "失王", "失网", "石望"],
            "惊讶": ["京讶", "精讶", "经讶", "惊压"],
            "意外": ["义外", "益外", "议外", "易外"],
            "奇怪": ["奇怪", "齐怪", "其怪", "期怪"],
            "正常": ["正长", "正常", "政常", "正场"],
            "异常": ["异长", "易常", "异场", "义常"],
            "普通": ["普同", "仆通", "铺通", "葡通"],
            "特殊": ["特殊", "特书", "特输", "特属"],
            "一般": ["一班", "移般", "义般", "议般"],
            
            # 生活用品类
            "东西": ["东细", "东西", "冬西", "东息"],
            "事情": ["事清", "世情", "事青", "是情"],
            "问题": ["问体", "文题", "问替", "问提"],
            "方法": ["方发", "房法", "芳法", "防法"],
            "办法": ["办发", "半法", "板法", "伴法"],
            "条件": ["条建", "跳件", "条见", "挑件"],
            "机会": ["鸡会", "机汇", "机回", "机惠"],
            "机遇": ["鸡遇", "机域", "机育", "机语"],
            "可能": ["可能", "科能", "课能", "刻能"],
            "能够": ["能构", "能狗", "能够", "能沟"],
            
            # 学习工作类
            "工作": ["公作", "工做", "攻作", "功作"],
            "学习": ["雪习", "学西", "学细", "血习"],
            "学生": ["雪生", "学声", "学升", "血生"],
            "老师": ["老是", "老师", "劳师", "老诗"],
            "同学": ["同雪", "童学", "通学", "桐学"],
            "同事": ["同是", "童事", "通事", "同世"],
            "朋友": ["朋有", "朋友", "蓬友", "鹏友"],
            "学校": ["雪校", "学校", "血校", "学效"],
            "公司": ["公思", "工司", "功司", "攻司"],
            "单位": ["单围", "丹位", "单维", "但位"],
            "部门": ["部们", "不门", "步门", "布门"],
            
            # 家庭关系类
            "父母": ["父木", "富母", "府母", "父亩"],
            "孩子": ["孩紫", "孩子", "骸子", "海子"],
            "儿子": ["而子", "尔子", "儿紫", "二子"],
            "女儿": ["女而", "女尔", "女二", "女儿"],
            "妻子": ["期子", "齐子", "其子", "妻紫"],
            "丈夫": ["丈服", "丈福", "帐夫", "张夫"],
            "兄弟": ["兄第", "胸弟", "雄弟", "熊弟"],
            "姐妹": ["姐美", "解妹", "街妹", "节妹"],
            "亲戚": ["亲期", "青戚", "亲齐", "亲其"],
            "邻居": ["邻据", "临居", "邻剧", "林居"],
            
            # 身体健康类
            "身体": ["身体", "神体", "身题", "身替"],
            "健康": ["健康", "见康", "建康", "坚康"],
            "生病": ["声病", "升病", "胜病", "省病"],
            "感冒": ["敢冒", "感帽", "感毛", "干冒"],
            "头疼": ["头腾", "投疼", "头藤", "头痛"],
            "发烧": ["发绕", "法烧", "发少", "发烧"],
            "咳嗽": ["刻嗽", "课嗽", "客嗽", "可嗽"],
            "治疗": ["制疗", "治聊", "至疗", "志疗"],
            "药物": ["药勿", "要物", "摇物", "姚物"],
            "医院": ["医源", "一院", "衣院", "易院"],
            
            # 交通出行类
            "交通": ["胶通", "教通", "叫通", "脚通"],
            "汽车": ["气车", "汽车", "起车", "齐车"],
            "火车": ["货车", "伙车", "或车", "活车"],
            "飞机": ["非机", "肥机", "费机", "飞鸡"],
            "地铁": ["地贴", "地铁", "底铁", "弟铁"],
            "公交": ["工交", "功交", "攻交", "公胶"],
            "出租": ["出组", "除租", "初租", "出主"],
            "开车": ["开车", "凯车", "开差", "开茶"],
            "坐车": ["做车", "作车", "座车", "左车"],
            "下车": ["夏车", "吓车", "下茶", "下差"],
            
            # 购物消费类
            "商店": ["伤店", "商点", "上店", "商典"],
            "超市": ["超是", "朝市", "潮市", "抄市"],
            "市场": ["世场", "是场", "市常", "师场"],
            "价格": ["架格", "假格", "价各", "家格"],
            "便宜": ["便议", "便易", "变宜", "遍宜"],
            "昂贵": ["昂归", "昂规", "昂贵", "昂跪"],
            "花钱": ["花前", "花钱", "华钱", "化钱"],
            "省钱": ["省前", "圣钱", "胜钱", "剩钱"],
            "付钱": ["付前", "负钱", "富钱", "副钱"],
            "找钱": ["召钱", "照钱", "招钱", "着钱"],
            
            # 娱乐休闲类
            "电视": ["电是", "电视", "电适", "电事"],
            "电影": ["电硬", "电应", "电迎", "电影"],
            "音乐": ["音月", "阴乐", "因乐", "音勒"],
            "游戏": ["有戏", "友戏", "又戏", "游细"],
            "运动": ["云动", "晕动", "孕动", "韵动"],
            "旅游": ["旅有", "旅游", "驴游", "履游"],
            "休息": ["修息", "秀息", "锈息", "宿息"],
            "放松": ["防松", "房松", "芳松", "方松"],
            "娱乐": ["鱼乐", "于乐", "愉乐", "余乐"],
            "享受": ["想受", "响受", "像受", "乡受"]
        }

        # 电视剧专业词汇大幅扩展
        self.tv_drama_terms = {
            # 影视制作类
            "主角": ["主脚", "朱角", "住角", "主交"],
            "配角": ["陪角", "配脚", "配交", "赔角"],
            "演员": ["言员", "眼员", "岩员", "延员"],
            "导演": ["道演", "倒演", "到演", "盗演"],
            "编剧": ["编据", "编局", "编剧", "便剧"],
            "制片": ["制篇", "制片", "制偏", "治片"],
            "监制": ["监制", "坚制", "建制", "检制"]
        }

    def post_process(self, text: str) -> str:
        """文本后处理主函数"""
        if not text:
            return text
            
        try:
            # 应用各种纠错规则
            text = self._correct_professional_terms(text)
            text = self._correct_polyphone_errors(text)
            text = self._correct_filler_words(text)
            text = self._add_punctuation(text)
            text = self._smart_sentence_segmentation(text)
            
            return text.strip()
        except Exception as e:
            logger.warning(f"文本后处理失败: {e}")
            return text

    def _correct_professional_terms(self, text: str) -> str:
        """纠正专业术语"""
        corrections_made = 0
        
        # 合并所有专业术语字典
        all_terms = {}
        if hasattr(self, 'professional_corrections'):
            all_terms.update(self.professional_corrections)
        if hasattr(self, 'tv_drama_terms'):
            all_terms.update(self.tv_drama_terms)
            
        for correct, wrong_list in all_terms.items():
            for wrong in wrong_list:
                if wrong in text:
                    text = text.replace(wrong, correct)
                    corrections_made += 1
                    
        if corrections_made > 0:
            logger.debug(f"专业术语纠正: {corrections_made} 处")
        return text

    def _correct_polyphone_errors(self, text: str) -> str:
        """纠正多音字错误"""
        # 简化的多音字纠错
        polyphone_map = {
            "银行": ["音行", "印行"],
            "重要": ["种要", "中要"],
            "处理": ["出理", "除理"],
            "数量": ["树量", "束量"]
        }
        
        corrections_made = 0
        for correct, wrong_list in polyphone_map.items():
            for wrong in wrong_list:
                if wrong in text:
                    text = text.replace(wrong, correct)
                    corrections_made += 1
                    
        if corrections_made > 0:
            logger.debug(f"多音字纠正: {corrections_made} 处")
        return text

    def _correct_filler_words(self, text: str) -> str:
        """纠正语气词和填充词"""
        if hasattr(self, 'filler_corrections'):
            corrections_made = 0
            for correct, wrong in self.filler_corrections.items():
                if wrong in text:
                    text = text.replace(wrong, correct)
                    corrections_made += 1
            
            if corrections_made > 0:
                logger.debug(f"语气词纠正: {corrections_made} 处")
        
        return text

    def _add_punctuation(self, text: str) -> str:
        """智能添加标点符号"""
        # 简化的标点添加
        text = re.sub(r'(\s+)', ' ', text)  # 规范化空格
        text = re.sub(r'([。！？])\s*([a-zA-Z\u4e00-\u9fff])', r'\1\n\2', text)  # 句末换行
        
        return text

    def _smart_sentence_segmentation(self, text: str) -> str:
        """智能断句"""
        # 基于语义的简单断句
        pause_markers = ['那么', '然后', '接着', '所以', '因此', '但是', '不过', '而且']
        
        for marker in pause_markers:
            text = text.replace(marker, f'，{marker}')
            
        return text

    def get_correction_stats(self, text: str) -> Dict[str, int]:
        """获取纠错统计"""
        stats = {
            'professional_terms': 0,
            'polyphone_errors': 0,
            'number_units': 0
        }
        
        # 简单统计
        if hasattr(self, 'professional_corrections'):
            for correct, wrong_list in self.professional_corrections.items():
                for wrong in wrong_list:
                    stats['professional_terms'] += text.count(wrong)
        
        return stats

    def analyze_text_quality(self, text: str) -> Dict[str, Any]:
        """分析文本质量"""
        stats = self.get_correction_stats(text)
        total_errors = sum(stats.values())
        total_chars = len(text)
        
        error_rate = (total_errors / total_chars * 100) if total_chars > 0 else 0
        quality_score = max(0, 100 - error_rate * 2)
        
        return {
            'quality_score': f"{quality_score:.1f}/100",
            'error_rate': f"{error_rate:.1f}",
            'error_statistics': {
                'sound_alike_errors': stats.get('polyphone_errors', 0),
                'professional_terms': stats.get('professional_terms', 0),
                'filler_words': 0
            },
            'recommendations': [
                "建议启用多模型融合以提高准确性",
                "可考虑使用更高质量的音频预处理"
            ] if error_rate > 5 else []
        }

    def add_custom_correction(self, correct_word: str, wrong_words: List[str]):
        """添加自定义纠错规则"""
        if not hasattr(self, 'custom_corrections'):
            self.custom_corrections = {}
        self.custom_corrections[correct_word] = wrong_words
        logger.info(f"已添加自定义纠错: {correct_word} <- {wrong_words}") "间制", "监支"],
            "制作": ["制做", "制作", "治作", "制昨"],
            "拍摄": ["拍射", "拍设", "派摄", "拍摄"],
            "剪辑": ["剪集", "减辑", "剪及", "见辑"],
            
            # 剧本内容类
            "剧本": ["据本", "剧奔", "剧本", "居本"],
            "剧情": ["据情", "举情", "剧清", "居情"],
            "情节": ["情结", "清洁", "情街", "青节"],
            "故事": ["古事", "股事", "故是", "顾事"],
            "台词": ["台词", "抬词", "台磁", "台辞"],
            "对白": ["对百", "对摆", "对白", "兑白"],
            "独白": ["毒白", "读白", "独百", "独摆"],
            "旁白": ["胖白", "旁百", "旁摆", "螃白"],
            "字幕": ["字母", "字幕", "子幕", "字木"],
            "片段": ["片断", "偏段", "篇段", "片端"],
            
            # 角色人物类
            "角色": ["脚色", "叫色", "角涩", "教色"],
            "人物": ["人勿", "任物", "人物", "仁物"],
            "性格": ["性各", "星格", "醒格", "性格"],
            "个性": ["各性", "个星", "革性", "歌性"],
            "形象": ["型象", "形像", "刑象", "行象"],
            "气质": ["气至", "起质", "气智", "器质"],
            "特点": ["特店", "特点", "特典", "特电"],
            "特色": ["特涩", "特色", "特社", "特设"],
            "风格": ["风各", "丰格", "枫格", "封格"],
            "魅力": ["魅力", "媚力", "美力", "妹力"],
            
            # 情感表达类
            "感情": ["敢情", "感清", "感青", "干情"],
            "爱情": ["爱清", "哀情", "爱青", "挨情"],
            "友情": ["友清", "有情", "友青", "由情"],
            "亲情": ["亲清", "青情", "亲青", "钦情"],
            "恋爱": ["连爱", "练爱", "恋爱", "链爱"],
            "喜欢": ["西欢", "稀欢", "喜换", "喜欢"],
            "讨厌": ["讨嫌", "掏厌", "讨严", "讨研"],
            "嫉妒": ["及妒", "急妒", "集妒", "记妒"],
            "怨恨": ["愿恨", "原恨", "缘恨", "源恨"],
            "愤怒": ["粉怒", "奋怒", "份怒", "分怒"],
            
            # 人际关系类
            "婚姻": ["昏姻", "婚银", "婚因", "混姻"],
            "结婚": ["结昏", "节婚", "接婚", "街婚"],
            "离婚": ["离昏", "理婚", "里婚", "立婚"],
            "夫妻": ["夫期", "夫齐", "夫其", "福妻"],
            "恋人": ["连人", "练人", "链人", "联人"],
            "情侣": ["清侣", "青侣", "情旅", "晴侣"],
            "暗恋": ["案恋", "暗练", "暗连", "暗链"],
            "表白": ["表百", "表摆", "标白", "彪白"],
            "求婚": ["球婚", "秋婚", "求昏", "邱婚"],
            "订婚": ["定婚", "订昏", "丁婚", "顶婚"],
            
            # 家庭场所类
            "家庭": ["家廷", "家停", "家亭", "加庭"],
            "房子": ["房紫", "方子", "房子", "防子"],
            "房间": ["房键", "房建", "房见", "房监"],
            "客厅": ["课厅", "客停", "客听", "客亭"],
            "卧室": ["我室", "卧是", "卧室", "握室"],
            "厨房": ["除房", "厨方", "厨房", "初房"],
            "浴室": ["欲室", "浴是", "浴室", "育室"],
            "阳台": ["羊台", "杨台", "洋台", "阳抬"],
            "花园": ["花元", "华园", "花圆", "化园"],
            "车库": ["车苦", "车哭", "车库", "车酷"],
            
            # 工作场所类
            "办公室": ["办公是", "办工室", "办功室", "半公室"],
            "会议室": ["会议是", "回忆室", "绘议室", "汇议室"],
            "公司": ["公思", "工司", "功司", "攻司"],
            "学校": ["学效", "雪校", "血校", "学校"],
            "医院": ["医源", "一院", "衣院", "易院"],
            "银行": ["音行", "印行", "阴行", "银航"],
            "警察局": ["警擦局", "警察剧", "精察局", "惊察局"],
            "法院": ["法源", "法院", "发院", "罚院"],
            "政府": ["正府", "政负", "整府", "证府"],
            "企业": ["期业", "齐业", "其业", "企叶"],
            
            # 公共场所类
            "餐厅": ["餐停", "惨厅", "餐听", "餐亭"],
            "酒店": ["酒店", "就店", "救店", "久店"],
            "宾馆": ["宾管", "滨馆", "宾官", "冰馆"],
            "商场": ["伤场", "商厂", "上场", "商常"],
            "超市": ["超是", "朝市", "潮市", "抄市"],
            "机场": ["鸡场", "急场", "及场", "集场"],
            "火车站": ["火车占", "货车站", "或车站", "活车站"],
            "汽车站": ["汽车占", "气车站", "起车站", "齐车站"],
            "公园": ["工园", "公元", "功园", "攻园"],
            "广场": ["广厂", "管场", "光场", "广常"],
            
            # 娱乐场所类
            "电影院": ["电硬院", "电影源", "电应院", "电迎院"],
            "剧院": ["据院", "剧源", "举院", "居院"],
            "音乐厅": ["音月厅", "阴乐厅", "因乐厅", "音勒厅"],
            "咖啡厅": ["咖非厅", "卡啡厅", "咖费厅", "咖飞厅"],
            "酒吧": ["酒八", "救吧", "久吧", "就吧"],
            "夜总会": ["叶总会", "夜宗会", "夜综会", "夜种会"],
            "健身房": ["建身房", "健神房", "坚身房", "见身房"],
            "游泳池": ["有泳池", "游咏池", "游永池", "游用池"],
            "网吧": ["忘吧", "网八", "王吧", "往吧"],
            "图书馆": ["图书管", "图书官", "图输管", "兔书馆"]
        }

        # 语气词和填充词纠正（扩展版）
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
            "嗯嗯嗯": "嗯",
            "哎哎哎": "哎",
            "唉唉唉": "唉",
            "咦咦咦": "咦",
            "哇哇哇": "哇",
            "哟哟哟": "哟",
            "咯咯咯": "咯",
            "啦啦啦": "啦"
        }

        # 新增：成语和固定搭配纠错
        self.idiom_corrections = {
            "一举两得": ["一举两的", "一举两地", "一举量得"],
            "一石二鸟": ["一是二鸟", "一石儿鸟", "一实二鸟"],
            "三心二意": ["三新二意", "三心儿意", "三辛二意"],
            "四面八方": ["四面八房", "是面八方", "四免八方"],
            "五湖四海": ["五户四海", "五湖是海", "五湖四孩"],
            "六神无主": ["六神误主", "六申无主", "六神无住"],
            "七上八下": ["七商八下", "七上八夏", "起上八下"],
            "八仙过海": ["八现过海", "八仙国海", "八仙过孩"],
            "九牛一毛": ["九牛一毛", "九牛移毛", "就牛一毛"],
            "十全十美": ["是全十美", "十权十美", "实全十美"],
            "百发百中": ["白发百中", "百法百中", "百发白中"],
            "千军万马": ["前军万马", "千君万马", "钱军万马"],
            "万水千山": ["晚水千山", "万税千山", "万水前山"],
            "心想事成": ["新想事成", "心像事成", "心想是成"],
            "马到成功": ["马道成功", "马到城功", "码到成功"],
            "龙飞凤舞": ["龙非凤舞", "龙飞风舞", "隆飞凤舞"],
            "虎头蛇尾": ["胡头蛇尾", "虎投蛇尾", "户头蛇尾"],
            "鸡犬不宁": ["鸡全不宁", "机犬不宁", "鸡犬布宁"],
            "画蛇添足": ["话蛇添足", "华蛇添足", "画射添足"],
            "守株待兔": ["守珠待兔", "受株待兔", "守株呆兔"],
            "亡羊补牢": ["忘羊补牢", "王羊补牢", "亡样补牢"],
            "杯弓蛇影": ["被弓蛇影", "杯功蛇影", "杯弓射影"],
            "井底之蛙": ["井地之蛙", "精底之蛙", "进底之蛙"],
            "狐假虎威": ["胡假虎威", "狐价虎威", "狐假胡威"],
            "画龙点睛": ["话龙点睛", "华龙点睛", "画隆点睛"],
            "滴水穿石": ["滴税穿石", "低水穿石", "滴水川石"],
            "铁杵成针": ["铁楚成针", "铁处成针", "铁杵城针"]
        }

        # 新增：专业术语纠错（医学、法律、金融等）
        self.professional_corrections = {
            # 医学术语
            "诊断": ["正断", "争断", "整断", "诊端"],
            "治疗": ["制疗", "治聊", "至疗", "志疗"],
            "手术": ["手数", "受术", "首术", "守术"],
            "药物": ["药勿", "要物", "摇物", "姚物"],
            "病症": ["病证", "病正", "病政", "病整"],
            "症状": ["政状", "正状", "争状", "整状"],
            "检查": ["检察", "见查", "检茶", "建查"],
            "化验": ["化言", "话验", "华验", "化研"],
            "康复": ["抗复", "康复", "康福", "慷复"],
            "预防": ["与防", "预房", "预坊", "玉防"],
            
            # 法律术语
            "法律": ["法绿", "法率", "法律", "发律"],
            "法官": ["法管", "法官", "发官", "法冠"],
            "律师": ["绿师", "率师", "律师", "律事"],
            "法院": ["法源", "法院", "发院", "罚院"],
            "判决": ["盘决", "判绝", "潘决", "叛决"],
            "证据": ["正据", "争据", "整据", "政据"],
            "诉讼": ["诉宋", "速讼", "苏讼", "诉送"],
            "合同": ["和同", "合童", "河同", "何同"],
            "契约": ["器约", "气约", "期约", "齐约"],
            "违约": ["围约", "为约", "维约", "喂约"],
            
            # 金融术语
            "银行": ["音行", "印行", "阴行", "银航"],
            "存款": ["存宽", "村款", "存款", "存快"],
            "贷款": ["贷宽", "代款", "带款", "戴款"],
            "利息": ["利西", "立息", "里息", "力息"],
            "投资": ["头资", "投姿", "偷资", "透资"],
            "股票": ["骨票", "古票", "股飘", "谷票"],
            "基金": ["基今", "机金", "即金", "及金"],
            "债券": ["债卷", "债劵", "债券", "债圈"],
            "保险": ["报险", "保显", "宝险", "包险"],
            "理财": ["理材", "里财", "立财", "力财"],
            
            # 科技术语
            "计算机": ["机算机", "计蒜机", "记算机", "积算机"],
            "网络": ["忘络", "网洛", "王络", "往络"],
            "软件": ["软键", "软见", "软建", "软监"],
            "硬件": ["应件", "硬键", "赢件", "营件"],
            "程序": ["成序", "城序", "承序", "程续"],
            "系统": ["细统", "西统", "系通", "洗统"],
            "数据": ["数具", "书据", "属据", "数剧"],
            "信息": ["心息", "新息", "信西", "信细"],
            "技术": ["记术", "技数", "即术", "及术"],
            "科学": ["课学", "科雪", "可学", "刻学"]
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
        # 移除多余的标点符号
        text = re.sub(r'[。]{2,}', '。', text)
        text = re.sub(r'[，]{2,}', '，', text)
        text = re.sub(r'[！]{2,}', '！', text)
        text = re.sub(r'[？]{2,}', '？', text)
        return text

    def _correct_sound_alike_errors(self, text: str) -> str:
        """纠正音对字不对的错误（增强版）"""
        # 统计纠正次数
        corrections_made = 0
        
        # 处理日常用语错误
        for correct, wrong_list in self.sound_alike_errors.items():
            for wrong in wrong_list:
                if wrong in text:
                    text = text.replace(wrong, correct)
                    corrections_made += 1
        
        # 处理电视剧专业词汇
        for correct, wrong_list in self.tv_drama_terms.items():
            for wrong in wrong_list:
                if wrong in text:
                    text = text.replace(wrong, correct)
                    corrections_made += 1
        
        # 处理成语和固定搭配
        for correct, wrong_list in self.idiom_corrections.items():
            for wrong in wrong_list:
                if wrong in text:
                    text = text.replace(wrong, correct)
                    corrections_made += 1
        
        # 处理专业术语
        for correct, wrong_list in self.professional_corrections.items():
            for wrong in wrong_list:
                if wrong in text:
                    text = text.replace(wrong, correct)
                    corrections_made += 1
        
        if corrections_made > 0:
            logger.debug(f"同音字纠正: {corrections_made} 处")
        
        return text

    def _correct_professional_terms(self, text: str) -> str:
        """纠正专业术语"""
        professional_terms = self.config.get("professional_terms", {})
        corrections_made = 0

        for correct_term, incorrect_terms in professional_terms.items():
            for incorrect_term in incorrect_terms:
                if incorrect_term in text:
                    text = text.replace(incorrect_term, correct_term)
                    corrections_made += 1
        
        if corrections_made > 0:
            logger.debug(f"专业术语纠正: {corrections_made} 处")
        
        return text

    def _correct_polyphones(self, text: str) -> str:
        """根据上下文纠正多音字（增强版）"""
        polyphone_corrections = self.config.get("polyphone_corrections", {})
        corrections_made = 0

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
            logger.debug(f"多音字纠正: {corrections_made} 处")
        
        return text

    def _process_numbers_and_units(self, text: str) -> str:
        """处理数字和单位（增强版）"""
        number_unit_corrections = self.config.get("number_unit_corrections", {})
        corrections_made = 0

        for correct_form, incorrect_forms in number_unit_corrections.items():
            for incorrect_form in incorrect_forms:
                if incorrect_form in text:
                    text = text.replace(incorrect_form, correct_form)
                    corrections_made += 1
        
        # 扩展的数字读音纠正
        digit_corrections = {
            "零": ["另", "令", "玲", "领"],
            "一": ["医", "衣", "以", "易"],
            "二": ["尔", "而", "儿", "耳"], 
            "三": ["伞", "散", "山", "闪"],
            "四": ["是", "死", "事", "世"],
            "五": ["我", "吴", "无", "午"],
            "六": ["留", "流", "刘", "柳"],
            "七": ["期", "齐", "其", "奇"],
            "八": ["吧", "把", "巴", "爸"],
            "九": ["就", "酒", "久", "救"],
            "十": ["是", "实", "石", "时"]
        }
        
        for correct, wrong_list in digit_corrections.items():
            for wrong in wrong_list:
                # 只在数字上下文中替换
                pattern = f'({wrong})([十百千万亿])'
                replacement = f'{correct}\\2'
                if re.search(pattern, text):
                    text = re.sub(pattern, replacement, text)
                    corrections_made += 1
                
                pattern = f'([十百千万亿])({wrong})'
                replacement = f'\\1{correct}'
                if re.search(pattern, text):
                    text = re.sub(pattern, replacement, text)
                    corrections_made += 1

        # 处理数字组合错误
        number_combinations = {
            "一十": ["医十", "衣十", "一是"],
            "二十": ["尔十", "而十", "二是"],
            "三十": ["伞十", "散十", "三是"],
            "四十": ["是十", "死十", "四是"],
            "五十": ["我十", "吴十", "五是"],
            "六十": ["留十", "流十", "六是"],
            "七十": ["期十", "齐十", "七是"],
            "八十": ["吧十", "把十", "八是"],
            "九十": ["就十", "酒十", "九是"],
            "一百": ["医百", "衣百", "一白"],
            "二百": ["尔百", "而百", "二白"],
            "三百": ["伞百", "散百", "三白"],
            "五百": ["我百", "吴百", "五白"],
            "一千": ["医千", "衣千", "一前"],
            "两千": ["量千", "亮千", "两前"],
            "三千": ["伞千", "散千", "三前"],
            "一万": ["医万", "衣万", "一晚"],
            "两万": ["量万", "亮万", "两晚"],
            "十万": ["是万", "时万", "实万"]
        }
        
        for correct, wrong_list in number_combinations.items():
            for wrong in wrong_list:
                if wrong in text:
                    text = text.replace(wrong, correct)
                    corrections_made += 1

        if corrections_made > 0:
            logger.debug(f"数字单位纠正: {corrections_made} 处")

        return text

    def _correct_filler_words(self, text: str) -> str:
        """纠正语气词和填充词（增强版）"""
        corrections_made = 0
        
        for wrong, correct in self.filler_corrections.items():
            if wrong in text:
                text = text.replace(wrong, correct)
                corrections_made += 1

        # 处理重复的语气词（更精确）
        repeated_patterns = [
            (r'([啊哦嗯额呃哈呵嘿咦哇哟咯啦])\1{2,}', r'\1'),
            (r'(那个){3,}', '那个'),
            (r'(这个){3,}', '这个'),
            (r'(就是){3,}', '就是'),
            (r'(然后){3,}', '然后'),
            (r'(对){4,}', '对'),
            (r'(是){4,}', '是'),
            (r'(好){4,}', '好')
        ]
        
        for pattern, replacement in repeated_patterns:
            if re.search(pattern, text):
                text = re.sub(pattern, replacement, text)
                corrections_made += 1
        
        # 处理重复的字词（超过2次重复）
        if re.search(r'(.)\1{3,}', text):
            text = re.sub(r'(.)\1{3,}', r'\1\1', text)
            corrections_made += 1

        if corrections_made > 0:
            logger.debug(f"语气词纠正: {corrections_made} 处")

        return text

    def _smart_sentence_segmentation(self, text: str) -> str:
        """智能断句（增强版）"""
        if not text:
            return text

        # 使用jieba分词
        words = list(jieba.cut(text))
        result = []

        # 扩展的语言标记词
        pause_words = ['那么', '然后', '接着', '于是', '所以', '因此', '另外', '还有', '而且', '并且', '同时', '此外', '总之', '最后', '最终', '首先', '其次', '再次', '最后', '综上所述']
        transition_words = ['但是', '不过', '然而', '可是', '只是', '只不过', '虽然', '尽管', '即使', '哪怕', '无论', '不管', '除非', '要不是', '假如', '如果']
        question_words = ['什么', '哪里', '怎么', '为什么', '怎样', '如何', '多少', '几', '吗', '呢', '啊', '哪个', '哪些', '何时', '何地', '何人']
        exclamation_words = ['太', '真', '好', '很', '非常', '特别', '极其', '相当', '十分', '格外', '特殊', '异常', '超级', '超']
        
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

            # 处理疑问句（增强版）
            if word in question_words:
                # 检查是否已经在句末
                remaining_words = words[i+1:]
                if not remaining_words or (len(remaining_words) <= 3 and all(w in ['吗', '呢', '啊', '的', '呀', '哇'] for w in remaining_words)):
                    if i == len(words) - 1 or (i < len(words) - 2 and words[i+1] in ['吗', '呢', '啊', '呀']):
                        result.append('？')
                        if i < len(words) - 1 and words[i+1] in ['吗', '呢', '啊', '呀']:
                            i += 1  # 跳过语气词

            # 处理感叹句（增强版）
            if word in exclamation_words and i < len(words) - 1:
                # 检查后面是否有形容词或副词
                next_words = words[i+1:i+4]
                if any(w in ['好', '棒', '厉害', '了不起', '不错', '完美', '精彩', '出色', '优秀', '杰出'] for w in next_words):
                    # 在句末添加感叹号
                    j = i + 1
                    while j < len(words) and words[j] not in sentence_enders:
                        j += 1
                    if j == len(words):  # 到了句末
                        result.append('！')

            i += 1

        text = ''.join(result)

        # 处理长句子，自动分段（改进版）
        sentences = re.split(r'([。！？])', text)
        processed_sentences = []

        for sentence in sentences:
            if sentence and sentence not in ['。', '！', '？']:
                # 如果句子太长（超过40个字符）且没有标点，添加逗号分隔
                if len(sentence) > 40 and '，' not in sentence:
                    # 在连词处分隔
                    sentence = re.sub(r'(然后|接着|于是|所以|因此|但是|不过|然而)', r'，\1', sentence)
                    sentence = re.sub(r'(而且|并且|同时|另外|还有|此外)', r'，\1', sentence)
                    sentence = re.sub(r'(首先|其次|再次|最后)', r'，\1', sentence)

                # 如果句子长度超过一定长度且没有标点，添加句号
                if len(sentence) > 20 and not sentence.endswith(('。', '！', '？', '，')):
                    sentence += '。'
            
            processed_sentences.append(sentence)

        return ''.join(processed_sentences)

    def _process_punctuation(self, text: str) -> str:
        """处理标点符号（增强版）"""
        # 移除多余的空格
        text = re.sub(r'\s+', '', text)

        # 处理连续的标点符号
        text = re.sub(r'[，]{2,}', '，', text)
        text = re.sub(r'[。]{2,}', '。', text)
        text = re.sub(r'[！]{2,}', '！', text)
        text = re.sub(r'[？]{2,}', '？', text)
        text = re.sub(r'[；]{2,}', '；', text)
        text = re.sub(r'[：]{2,}', '：', text)

        # 处理标点符号前的空格
        text = re.sub(r'\s+([，。！？；：])', r'\1', text)

        # 修正标点符号的使用（增强版）
        # 疑问句应该以问号结尾
        question_patterns = [
            (r'([什么哪里怎么为什么怎样如何多少几哪个哪些何时何地何人])([^？]*?)([。！])', r'\1\2？'),
            (r'(吗|呢|啊|呀|哇)([。！])', r'\1？'),
            (r'(是不是|对不对|行不行|好不好|要不要)([。！])', r'\1？')
        ]
        
        for pattern, replacement in question_patterns:
            text = re.sub(pattern, replacement, text)
        
        # 感叹句处理（增强版）
        exclamation_patterns = [
            (r'([太真好很非常特别极其相当十分格外超级超])([^！]*?)([。？])', r'\1\2！'),
            (r'(哇|呀|啊|哎呀|天哪|我的天)([。？])', r'\1！'),
            (r'(太棒了|太好了|真厉害|真不错|太精彩了)([。？])', r'\1！')
        ]
        
        for pattern, replacement in exclamation_patterns:
            text = re.sub(pattern, replacement, text)

        # 确保句子有合适的结尾
        if text and not text.endswith(('。', '！', '？')):
            # 如果最后是逗号，改为句号
            if text.endswith('，'):
                text = text[:-1] + '。'
            else:
                # 根据语境决定标点（增强版）
                last_10_chars = text[-15:] if len(text) > 15 else text
                
                # 检查疑问词
                question_indicators = ['什么', '怎么', '为什么', '吗', '呢', '啊', '哪', '多少', '几', '何']
                if any(qw in last_10_chars for qw in question_indicators):
                    text += '？'
                # 检查感叹词
                elif any(ew in last_10_chars for ew in ['太', '真', '好', '很', '非常', '哇', '呀']):
                    text += '！'
                # 检查结论性词汇
                elif any(cw in last_10_chars for cw in ['总之', '最后', '因此', '所以', '综上']):
                    text += '。'
                else:
                    text += '。'

        return text

    def post_process(self, text: str) -> str:
        """主要的文本后处理方法（增强版）"""
        if not text or not text.strip():
            return text

        original_text = text
        
        logger.debug(f"开始文本后处理，原始长度: {len(text)} 字符")
        
        # 1. 基础清理
        text = self._clean_text(text)

        # 2. 纠正音对字不对的错误（包括成语、专业术语）
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
            logger.info(f"文本后处理完成，共修正 {changes} 处，处理后长度: {len(text)} 字符")

        return text.strip()

    def _count_changes(self, original: str, processed: str) -> int:
        """统计文本变化数量"""
        if len(original) == 0 and len(processed) == 0:
            return 0
        
        # 使用编辑距离算法计算更精确的变化数
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        return levenshtein_distance(original, processed)

    def get_correction_stats(self, text: str) -> Dict[str, int]:
        """获取纠错统计信息（增强版）"""
        stats = {
            "sound_alike_errors": 0,
            "professional_terms": 0,
            "polyphone_errors": 0,
            "number_units": 0,
            "filler_words": 0,
            "idiom_errors": 0,
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
        for wrong, correct in self.filler_corrections.items():
            stats["filler_words"] += text.count(wrong)

        # 统计成语错误
        for correct, wrong_list in self.idiom_corrections.items():
            for wrong in wrong_list:
                stats["idiom_errors"] += text.count(wrong)

        return stats

    def add_custom_correction(self, correct_word: str, wrong_words: List[str], category: str = "custom"):
        """添加自定义纠错词汇（增强版）"""
        if category == "sound_alike":
            self.sound_alike_errors[correct_word] = wrong_words
        elif category == "tv_drama":
            self.tv_drama_terms[correct_word] = wrong_words
        elif category == "filler":
            for wrong in wrong_words:
                self.filler_corrections[wrong] = correct_word
        elif category == "idiom":
            self.idiom_corrections[correct_word] = wrong_words
        elif category == "professional":
            self.professional_corrections[correct_word] = wrong_words
        
        logger.info(f"添加自定义纠错词汇 ({category}): {correct_word} <- {wrong_words}")

    def analyze_text_quality(self, text: str) -> Dict[str, Any]:
        """分析文本质量"""
        stats = self.get_correction_stats(text)
        
        # 计算错误率
        total_errors = sum([stats[key] for key in stats if key != "total_chars"])
        error_rate = total_errors / max(stats["total_chars"], 1) * 100
        
        # 计算质量评分
        if error_rate <= 2:
            quality_score = "优秀"
        elif error_rate <= 5:
            quality_score = "良好"
        elif error_rate <= 10:
            quality_score = "一般"
        else:
            quality_score = "较差"
        
        return {
            "error_statistics": stats,
            "error_rate": round(error_rate, 2),
            "quality_score": quality_score,
            "recommendations": self._generate_text_recommendations(stats)
        }

    def _generate_text_recommendations(self, stats: Dict[str, int]) -> List[str]:
        """生成文本优化建议"""
        recommendations = []
        
        if stats["sound_alike_errors"] > 5:
            recommendations.append("建议启用高级同音字纠错功能")
        
        if stats["professional_terms"] > 3:
            recommendations.append("建议补充领域专业词汇库")
        
        if stats["filler_words"] > 10:
            recommendations.append("检测到较多口语化表达，建议进行语音清理")
        
        if stats["number_units"] > 2:
            recommendations.append("建议启用数字单位智能纠正")
        
        if stats["idiom_errors"] > 1:
            recommendations.append("建议启用成语固定搭配纠错")
        
        return recommendations
