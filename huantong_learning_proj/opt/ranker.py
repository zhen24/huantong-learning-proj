import logging

import attr
import torch
import cn2an

from rankerdog.data import create_bert_tokenizer, bert_tokenize_pair
from rankerdog.model import RankerModel

from textdog.lexicon.normalize import normalize
from retrievaldog.retriever import (
    rule_based_split,
    LEVEL_ONE_SUFFIX_SPLITS,
    LEVEL_TWO_SUFFIX_SPLITS,
    LEVEL_THREE_SUFFIX_SPLITS,
)

logger = logging.getLogger(__name__)


def ac_conditional_add(suffix_splits, field, name, texts):
    if not field:
        return

    field = normalize(field.strip())
    if not field:
        return

    group = rule_based_split(field, suffix_splits)
    prefix = group[0] if group else field
    if prefix not in name:
        logger.debug(f'prefix={prefix} not found, inject field={field} to texts.')
        texts.append(field)


def batch_to_device(batch, device):
    return {
        key: val.to(device, non_blocking=True) if torch.is_tensor(val) else val
        for key, val in batch.items()
    }


CHINESE_NUM_CHARS = {
    '一',
    '二',
    '三',
    '四',
    '五',
    '六',
    '七',
    '八',
    '九',
    '十',
}

CHINESE_NUM_CHARS_EXT = {
    *CHINESE_NUM_CHARS,
    '万',
    '两',
    '亿',
    '仟',
    '仨',
    '伍',
    '佰',
    '千',
    '叁',
    '壹',
    '幺',
    '拾',
    '捌',
    '柒',
    '玖',
    '百',
    '肆',
    '贰',
    '陆',
    '零',
}

CHINESE_NUM_TO_ARABIC_TO_DIRECT_MAPPING = {
    '零': '0',
    '一': '1',
    '二': '2',
    '三': '3',
    '四': '4',
    '五': '5',
    '六': '6',
    '七': '7',
    '八': '8',
    '九': '9',
    '十': '10',
}

STORE_NOISE_RELU = {
    "　": "",
    " ": "",
    "\t": "",
    "#": "",
    "¤": "",
    "%": "",
    "?": "",
    "~": "",
    "、": "",
    "。": "",
    "ˉ": "",
    "ˇ": "",
    "〃": "",
    "々": "",
    "—": "",
    "～": "",
    "‖": "",
    "…": "",
    "‘": "",
    "’": "",
    "“": "",
    "”": "",
    "∥": "",
    "×": "",
    "÷": "",
    "∧": "",
    "∨": "",
    "∈": "",
    "∷": "",
    "√": "",
    "⊥": "",
    "∠": "",
    "⌒": "",
    "⊙": "",
    "∫": "",
    "∮": "",
    "≈": "",
    "∽": "",
    "∝": "",
    "∞": "",
    "♂": "",
    "♀": "",
    "′": "",
    "＄": "",
    "￠": "",
    "￡": "",
    "§": "",
    ".": "",
    "☆": "",
    "★": "",
    "◎": "",
    "◇": "",
    "◆": "",
    "□": "",
    "■": "",
    "△": "",
    "▲": "",
    "※": "",
    "→": "",
    "←": "",
    "↑": "",
    "●": "",
    "↓": "",
    "〓": "",
    "！": "",
    "＃": "",
    "￥": "",
    "％": "",
    "＆": "",
    "＊": "",
    "＋": "",
    "，": "",
    "－": "",
    "．": "",
    "／": "",
    "：": "",
    "；": "",
    "？": "",
    "＠": "",
    "Ｙ": "",
    "＼": "",
    "｜": "",
    "￣": "",
    "︱": "",
    "︴": "",
    "ˊ": "",
    "ˋ": "",
    "–": "",
    "‥": "",
    "‵": "",
    "℅": "",
    "↖": "",
    "↗": "",
    "↘": "",
    "↙": "",
    "∕": "",
    "∣": "",
    "▼": "",
    "▽": "",
    "⊕": "",
    "〒": "",
    "〝": "",
    "〞": "",
    "㊣": "",
    "︰": "",
    "﹉": "",
    "﹏": "",
    "﹐": "",
    "﹑": "",
    "﹒": "",
    "﹔": "",
    "﹕": "",
    "﹖": "",
    "﹗": "",
    "﹟": "",
    "﹠": "",
    "﹡": "",
    "﹢": "",
    "﹣": "",
    "﹦": "",
    "﹨": "",
    "=": "",
    "+": "",
    "-": "",
    "_": "",
    "/": "",
    ",": "",
    ">": "",
    "<": "",
    "'": "",
    "\"": "",
    ":": "",
    ";": "",
    "|": "",
    "\\": "",
    "!": "",
    "@": "",
    "$": "",
    "^": "",
    "&": "",
    "*": "",
    "`": "",
    "x0000": "",
    "Ⅰ": "",
    "Ⅱ": "",
    "Ⅲ": "",
    "Ⅶ": "",
    "Ⅵ": "",
    "Ⅷ": "",
    "Ⅸ": "",
    "Ⅺ": "",
    "Ⅻ": "",
    "Ⅳ": "",
    "Ⅴ": "",
    "Ⅹ": "",
    "〔": "(",
    "〈": "(",
    "《": "(",
    "「": "(",
    "『": "(",
    "〖": "(",
    "【": "(",
    "（": "(",
    "＜": "(",
    "｛": "(",
    "﹙": "(",
    "﹛": "(",
    "﹝": "(",
    "﹤": "(",
    "[": "(",
    "{": "(",
    "〕": ")",
    "〉": ")",
    "》": ")",
    "」": ")",
    "』": ")",
    "〗": ")",
    "】": ")",
    "）": ")",
    "＞": ")",
    "｝": ")",
    "﹚": ")",
    "﹜": ")",
    "﹞": ")",
    "﹥": ")",
    "]": ")",
    "}": ")",
    "ⅰ": "1",
    "ⅱ": "2",
    "ⅲ": "3",
    "⑴": "1",
    "⑵": "2",
    "⑶": "3",
    "⑷": "4",
    "⑸": "5",
    "⑹": "6",
    "⑺": "7",
    "⑻": "8",
    "⑼": "9",
    "①": "1",
    "②": "2",
    "③": "3",
    "④": "4",
    "⑤": "5",
    "⑥": "6",
    "⑦": "7",
    "⑧": "8",
    "⑨": "9",
    "１": "1",
    "２": "2",
    "３": "3",
    "4": "4",
    "５": "5",
    "６": "6",
    "７": "7",
    "８": "8",
    "９": "9",
    "０": "0",
    "○": "0",
    "O": "0",
    "〇": "0",
    "中西药房": "",
    "西药房": "",
    "中药房": "",
    "医药": "",
    "零售": "",
    "药库": "",
    "药品": "",
    "药店": "店",
    "分店": "店",
    "地段医院": "医院",
    "医学院附属第": "附属第",
    "医院公司": "医院",
    "医院门诊药房": "医院",
    "医院门诊部": "医院",
    "医院门诊": "医院",
    "医学院附属医院": "附属医院",
    "医疗集团人民医院": "人民医院",
    "0TC": "",
    "有限责任公司": "",
    "有限公司": "",
    "连锁": "",
    "普通合伙": "",
    "有限合伙": "",
    "销售一部": "",
    "销售二部": "",
    "非一体化": "",
    "一体化": "",
    "必须按批号": "",
    "正常销售": "",
    "查询不到": "",
    "处方部": "",
    "药师帮": "",
    "出货单": "",
    "冲销": "",
    "终端": "",
    "军队": "部队",
    "马路对面": "",
    "对面": "",
    "交叉口": "",
    "交汇处": "",
    "附近": "",
    "旁边": "",
    "郊区": "",
    "东路": "东%",
    "西路": "西%",
    "南路": "南%",
    "北路": "北%",
    "中路": "",
    "环路": "",
    "东街": "东%",
    "西街": "西%",
    "南街": "南%",
    "北街": "北%",
    "中街": "",
    "大街": "",
    "省": "",
    "市": "",
    "区": "",
    "县": "",
    "路": "",
    "街": "",
    "弄": "",
    "单元": "",
    "幢": "",
    "栋": "",
    "层": "",
    "号": ""
}


def chunk_by_chinese_num(text):
    chunks = []
    chinese_num_chunk_indices = []
    begin = 0
    while begin < len(text):
        # Not startswith space.
        assert not text[begin].isspace()

        # Find a chunk of chinese number.
        end = begin
        while end < len(text) and (text[end].isspace() or text[end] in CHINESE_NUM_CHARS_EXT):
            end += 1

        if end == begin:
            # Not a chinese number chunk.
            while end < len(text) \
                    and (text[end].isspace() or text[end] not in CHINESE_NUM_CHARS_EXT):
                end += 1
            chunks.append(text[begin:end])
        else:
            # Is a chinse chunk.
            chunks.append(text[begin:end])
            chinese_num_chunk_indices.append(len(chunks) - 1)

        begin = end

    return chunks, chinese_num_chunk_indices


def normalize_chinese_num_chunk(text):
    raw_text = text
    text = ''.join(text.split())

    if len(text) == 1:
        # Single character case.
        if text in CHINESE_NUM_TO_ARABIC_TO_DIRECT_MAPPING:
            return CHINESE_NUM_TO_ARABIC_TO_DIRECT_MAPPING[text]
        else:
            return raw_text
    else:
        # Multiple characters case.
        try:
            return str(cn2an.cn2an(text, 'smart'))
        except (ValueError, KeyError):
            # Failed to convert, try to map directly.
            direct_mapping_chars = []
            for char in text:
                if char in CHINESE_NUM_TO_ARABIC_TO_DIRECT_MAPPING:
                    direct_mapping_chars.append(CHINESE_NUM_TO_ARABIC_TO_DIRECT_MAPPING[char])
                else:
                    # Failed, fall back.
                    return raw_text
            return ''.join(direct_mapping_chars)


def ranker_normalize(name):
    name = normalize(name.strip())
    chunks, chinese_num_chunk_indices = chunk_by_chinese_num(name)
    for idx in chinese_num_chunk_indices:
        chunks[idx] = normalize_chinese_num_chunk(chunks[idx])
    return ''.join(chunks)


def generate_loc_chunk(org, loc_tag, prefix):
    loc = getattr(org, loc_tag)
    if loc:
        return prefix + loc
    else:
        return prefix + '?'


def generate_text_pair(upstream_org, query, candidate_org):
    left_chunks = [
        generate_loc_chunk(upstream_org, 'province', ':'),
        # generate_loc_chunk(upstream_org, 'city'),
        # generate_loc_chunk(upstream_org, 'county'),
        '#',
        query,
    ]
    left_text = ''.join(left_chunks)
    left_text = ranker_normalize(left_text)

    right_chunks = [
        generate_loc_chunk(candidate_org, 'province', ':'),
        generate_loc_chunk(candidate_org, 'city', ':'),
        generate_loc_chunk(candidate_org, 'county', ':'),
        '#',
        candidate_org.name,
    ]
    right_text = ''.join(right_chunks)
    right_text = ranker_normalize(right_text)

    return left_text, right_text


def debug():
    chunks, chinese_num_chunk_indices = \
        chunk_by_chinese_num("襄阳千千天济大药房连锁有限公司暨襄樊天济大药房连锁公司一二四店")
    print(chunks, chinese_num_chunk_indices)

    texts = [
        '千千',
        '一二四',
        '一百二十四四',
    ]
    for text in texts:
        print(text, normalize_chinese_num_chunk(text))

    print(ranker_normalize("天济大药房连锁有限公司六百一十六分店"))
    print(ranker_normalize("襄阳天济大药房连锁有限公司襄阳三百零五店"))


@torch.no_grad()
def ranker_predict(
    model,
    bert_tokenizer,
    device,
    query,
    retrieved_orgs,
    apply_sigmoid=False,
):
    logger.debug('in ranker_predict')

    # Preprocess.
    query = ranker_normalize(query.strip())
    logger.debug(f'query={query}')

    text_pairs = []
    for retrieved_org in retrieved_orgs:

        name = ranker_normalize(retrieved_org.name.strip())
        texts = []
        ac_conditional_add(
            suffix_splits=LEVEL_ONE_SUFFIX_SPLITS,
            field=retrieved_org.province,
            name=name,
            texts=texts,
        )
        ac_conditional_add(
            suffix_splits=LEVEL_TWO_SUFFIX_SPLITS,
            field=retrieved_org.city,
            name=name,
            texts=texts,
        )
        ac_conditional_add(
            suffix_splits=LEVEL_THREE_SUFFIX_SPLITS,
            field=retrieved_org.county,
            name=name,
            texts=texts,
        )
        texts.append(name)

        cand_text = ''.join(texts)
        logger.debug(f'name={name}, cand_text={cand_text}')

        text_pairs.append((query, cand_text))

    # Forward.
    batch = bert_tokenize_pair(bert_tokenizer, text_pairs)
    batch = batch_to_device(batch, device)

    logits = model(
        input_ids=batch['input_ids'],
        token_type_ids=batch['token_type_ids'],
        attention_mask=batch['attention_mask'],
    )
    if apply_sigmoid:
        logits = torch.sigmoid(logits)

    indices = torch.argsort(logits, descending=True)
    logits = logits.cpu().tolist()

    sorted_retrieved_orgs = []
    sorted_logits = []
    for idx in indices:
        sorted_retrieved_orgs.append(retrieved_orgs[idx])
        sorted_logits.append(logits[idx])

    return sorted_retrieved_orgs, sorted_logits


@torch.no_grad()
def ranker_predict_20210715(
    model,
    bert_tokenizer,
    device,
    upstream_pg_org,
    query,
    retrieved_orgs,
    apply_sigmoid=False,
):
    logger.debug('in ranker_predict')

    # Preprocess.
    logger.debug(f'query={query}')

    text_pairs = []
    for retrieved_org in retrieved_orgs:
        text_pairs.append(generate_text_pair(upstream_pg_org, query, retrieved_org))

    # Forward.
    batch = bert_tokenize_pair(bert_tokenizer, text_pairs)
    batch = batch_to_device(batch, device)

    logits = model(
        input_ids=batch['input_ids'],
        token_type_ids=batch['token_type_ids'],
        attention_mask=batch['attention_mask'],
    )
    if apply_sigmoid:
        logits = torch.sigmoid(logits)

    indices = torch.argsort(logits, descending=True)
    logits = logits.cpu().tolist()

    sorted_retrieved_orgs = []
    sorted_logits = []
    for idx in indices:
        sorted_retrieved_orgs.append(retrieved_orgs[idx])
        sorted_logits.append(logits[idx])

    return sorted_retrieved_orgs, sorted_logits


@attr.s
class RankerConfig:
    bert_pretrained_folder = attr.ib()
    state_dict_file = attr.ib()
    device = attr.ib()
    # Classifier.
    classifier_top_k = attr.ib()
    classifier_thr = attr.ib()


def load_ranker_model(ranker_config):
    model = RankerModel(
        bert_pretrained_folder=ranker_config.bert_pretrained_folder,
        bert_from_config=True,
        bert_torchscript=False,
    )
    model = model.to(ranker_config.device)

    state_dict = torch.load(ranker_config.state_dict_file, map_location=ranker_config.device)
    state_dict = state_dict['model']

    model.load_state_dict(state_dict)
    model.eval()

    bert_tokenizer = create_bert_tokenizer(
        ranker_config.bert_pretrained_folder,
        512,
    )

    return model, bert_tokenizer
