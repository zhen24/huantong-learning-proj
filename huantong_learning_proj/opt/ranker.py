import logging
from typing import Union, Sequence, Any

import attr
import cn2an
import torch
from rankerdog.data import create_bert_tokenizer, bert_tokenize_pair
from rankerdog.model import RankerModel
from textdog.lexicon.normalize import normalize
from tool.train1 import train
from transformers import PreTrainedTokenizerBase, BertTokenizer

from huantong_learning_proj.db import PgOrganization
from huantong_learning_proj.opt.retriever import RetrievedOrg

logger = logging.getLogger(__name__)


def batch_to_device(batch: Any, device: str):
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


def chunk_by_chinese_num(text: str):
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


def normalize_chinese_num_chunk(text: str):
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


def ranker_normalize(name: str):
    name = normalize(name.strip())
    chunks, chinese_num_chunk_indices = chunk_by_chinese_num(name)
    for idx in chinese_num_chunk_indices:
        chunks[idx] = normalize_chinese_num_chunk(chunks[idx])
    return ''.join(chunks)


def generate_loc_chunk(org: Union[PgOrganization, RetrievedOrg], loc_tag: str, prefix: str):
    loc = getattr(org, loc_tag)
    if loc:
        return prefix + loc
    else:
        return prefix + '?'


def generate_text_pair(
    upstream_org: PgOrganization,
    query: str,
    candidate_org: RetrievedOrg,
):
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


@torch.no_grad()
def ranker_predict_20210715(
    model: RankerModel,
    bert_tokenizer: Union[PreTrainedTokenizerBase, BertTokenizer],
    device: str,
    upstream_pg_org: PgOrganization,
    query: str,
    retrieved_orgs: Sequence[RetrievedOrg],
    apply_sigmoid: bool = False,
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
        sorted_retrieved_orgs.append(retrieved_orgs[idx])  # type: ignore
        sorted_logits.append(logits[idx])

    return sorted_retrieved_orgs, sorted_logits


@attr.s
class RankerConfig:
    bert_pretrained_folder: str = attr.ib()
    state_dict_file: str = attr.ib()
    device: str = attr.ib()
    # Classifier.
    classifier_top_k: int = attr.ib()
    classifier_thr: float = attr.ib()


def load_ranker_model(ranker_config: RankerConfig):
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


def run_training(
    bert_pretrained_folder: str,
    train_dataset_folder: str,
    dev_dataset_folder: str,
    output_folder: str,
    device: str = 'cpu',
):
    train(
        bert_pretrained_folder,
        train_dataset_folder,
        dev_dataset_folder,
        output_folder,
        device,
    )
