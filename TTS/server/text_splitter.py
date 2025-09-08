"""
文本分段处理器 - 解决XTTS长文本不稳定问题
使用pysbd专业分句库，根据XTTS内部机制优化：
- 中文: 限制80字符 (XTTS内部按3字节/字符处理)
- 英文: 限制250字符
"""

import re
import pysbd
from typing import List, Tuple


class TextSplitter:
    def __init__(self, max_length: int = 250):
        """
        初始化文本分割器
        
        Args:
            max_length: 默认最大长度（英文）
        """
        self.max_length_en = min(max_length, 250)  # 英文最大250
        self.max_length_zh = 80  # 中文最大80（对应240字节）
        
    def detect_language(self, text: str, hint: str = 'auto') -> str:
        """
        检测文本主要语言
        
        Args:
            text: 输入文本
            hint: 语言提示 ('zh', 'en', 'auto')
        
        Returns:
            'zh' 或 'en'
        """
        if hint != 'auto':
            return hint
            
        # 简单检测：中文字符比例
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text.strip())
        
        if total_chars == 0:
            return 'en'
            
        chinese_ratio = chinese_chars / total_chars
        return 'zh' if chinese_ratio > 0.3 else 'en'
    
    def smart_split(self, text: str, language_hint: str = 'auto') -> List[Tuple[str, str]]:
        """
        智能分段算法 - 使用pysbd专业分句
        
        Args:
            text: 输入文本
            language_hint: 语言提示 ('zh', 'en', 'auto')
            
        Returns:
            List[(segment_text, language)] - 分段结果和语言
        """
        # 检测语言
        lang = self.detect_language(text, language_hint)
        # 支持 zh, zh-cn, zh-tw 等中文变体
        is_chinese = lang.startswith('zh')
        max_length = self.max_length_zh if is_chinese else self.max_length_en
        
        print(f" > Text splitting using pysbd: {len(text)} chars, lang={lang}, max={max_length}")
        
        # 如果文本已经够短，直接返回
        if len(text) <= max_length:
            return [(text, lang)]
        
        # 使用pysbd分句
        try:
            # pysbd语言代码映射
            pysbd_lang = lang if lang in ['en', 'zh'] else 'en'
            segmenter = pysbd.Segmenter(language=pysbd_lang, clean=False)
            sentences = segmenter.segment(text)
        except Exception as e:
            print(f" > WARNING: pysbd failed ({e}), using fallback")
            sentences = self.fallback_split(text, lang)
        
        print(f" > pysbd split into {len(sentences)} sentences")
        
        # 合并短句，分割长句
        segments = self.merge_sentences(sentences, max_length, lang)
        
        # 修复引号平衡
        segments = self.fix_quote_balance(segments)
        
        # 输出结果
        print(f" > Final result: {len(segments)} segments")
        for i, seg in enumerate(segments):
            preview = seg[:50] + "..." if len(seg) > 50 else seg
            print(f" >   Segment {i+1} ({len(seg)} chars): {preview}")
            
        return [(seg, lang) for seg in segments]
    
    def merge_sentences(self, sentences: List[str], max_length: int, lang: str) -> List[str]:
        """
        合并短句，分割长句
        """
        segments = []
        current = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 单句太长，需要进一步分割
            if len(sentence) > max_length:
                print(f" >   Long sentence ({len(sentence)} chars), splitting...")
                parts = self.split_long_sentence(sentence, max_length, lang)
                for part in parts:
                    if len(current + part) > max_length and current:
                        segments.append(current)
                        current = part
                    else:
                        # 中文不需要空格，英文需要
                        if lang.startswith('zh'):
                            current = current + part if current else part
                        else:
                            current = current + " " + part if current else part
            else:
                # 检查合并
                if lang.startswith('zh'):
                    test_merge = current + sentence if current else sentence
                else:
                    test_merge = current + " " + sentence if current else sentence
                
                if len(test_merge) > max_length and current:
                    segments.append(current)
                    current = sentence
                else:
                    current = test_merge
        
        if current:
            segments.append(current)
        
        return segments
    
    def split_long_sentence(self, sentence: str, max_length: int, lang: str) -> List[str]:
        """
        分割过长的句子
        """
        parts = []
        
        # 按逗号分割
        if lang.startswith('zh'):
            comma_parts = re.split(r'([，；、])', sentence)
        else:
            comma_parts = re.split(r'([,;])', sentence)
        
        # 重组包含标点的部分
        current = ""
        for i, part in enumerate(comma_parts):
            if not part.strip():
                continue
                
            if len(current + part) > max_length and current:
                parts.append(current)
                current = part
            else:
                current = current + part
        
        if current:
            parts.append(current)
        
        # 如果还是太长，按词/字符分割
        final_parts = []
        for part in parts:
            if len(part) > max_length:
                if lang.startswith('zh'):
                    # 中文按字符分割
                    for i in range(0, len(part), max_length):
                        final_parts.append(part[i:i+max_length])
                else:
                    # 英文按词分割
                    words = part.split()
                    current_part = ""
                    for word in words:
                        if len(current_part + " " + word) > max_length and current_part:
                            final_parts.append(current_part)
                            current_part = word
                        else:
                            current_part = current_part + " " + word if current_part else word
                    if current_part:
                        final_parts.append(current_part)
            else:
                final_parts.append(part)
        
        return final_parts
    
    def fix_quote_balance(self, segments: List[str]) -> List[str]:
        """
        修复引号平衡问题
        """
        if not segments:
            return segments
        
        fixed = []
        
        for i, seg in enumerate(segments):
            quote_count = seg.count('"')
            
            if quote_count % 2 == 1:
                # 奇数引号，需要修复
                print(f" >   Segment {i+1} has {quote_count} quotes (unbalanced)")
                
                # 检查结尾
                if seg.rstrip().endswith('"'):
                    # 结尾有引号，可能是开启引号
                    if i + 1 < len(segments) and not segments[i+1].lstrip().startswith('"'):
                        segments[i+1] = '"' + segments[i+1]
                        print(f" >     Added opening quote to segment {i+2}")
                else:
                    # 结尾没引号，添加闭合引号
                    seg = seg + '"'
                    print(f" >     Added closing quote to segment {i+1}")
            
            fixed.append(seg)
        
        return fixed
    
    def fallback_split(self, text: str, lang: str) -> List[str]:
        """
        备用分割方法（当pysbd不可用时）
        """
        if lang.startswith('zh'):
            # 中文按句号分割
            sentences = re.split(r'([。！？]+)', text)
        else:
            # 英文按句号分割
            sentences = re.split(r'([.!?]+)', text)
        
        # 重组句子
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i].strip()
            punct = sentences[i + 1] if i + 1 < len(sentences) else ""
            if sentence:
                result.append(sentence + punct)
        
        return result