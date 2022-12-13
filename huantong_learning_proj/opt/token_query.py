# -*- coding: utf-8 -*-
import logging
from typing import Sequence, Optional

import iolite as io
from textdog.lexicon.normalize import normalize
from textdog.token.ltp import LtpTokenizerConfig, LtpTokenizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s: [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def merge_words(words: Sequence[str]):
    if '药房' in words or '药店' in words:
        new_words = []
        idx = 0
        num_words = len(words)
        while idx < num_words:
            next_idx = idx + 1
            if (
                words[idx].endswith('大') and next_idx < num_words
                and words[next_idx] in {'药房', '药店'}
            ):
                front_part = words[idx][:-1]
                if front_part:
                    new_words.append(front_part)
                new_words.append('大' + words[idx + 1])
                idx += 2
                continue
            new_words.append(words[idx])
            idx += 1
        return new_words
    return words


def gen_ltp_tokens(
    ltp_tokenizer_config: LtpTokenizerConfig,
    input_csv: str,
    target_col: str,
    gen_col: Optional[str] = None,
    extend_csv: bool = False,
):
    ltp_tokenizer = LtpTokenizer(ltp_tokenizer_config)
    rows = list(io.read_csv_lines(input_csv, skip_header=True, to_dict=True))
    texts = [normalize(row[target_col]) for row in rows]
    logger.info(f'Processing {len(texts)} text ...')

    if gen_col is None:
        gen_col = f'{target_col}_tokens'
    for idx, tokens in enumerate(tqdm(ltp_tokenizer.batch_tokenize(texts))):
        text = texts[idx]
        merged_tokens = merge_words(tokens)
        # logger.info({'text': text, 'tokens': merged_tokens})

        if not extend_csv:
            yield {'text': text, 'tokens': merged_tokens}
        else:  # Extend csv table.
            row = rows[idx]
            row[gen_col] = '|'.join(merged_tokens)  # query_name_tokens
            row['request_at'] = '2021-06-15 00:00:00'
            row['response_at'] = '2021-06-15 00:00:00'
            yield row


def debug_tokenize(
    input_csv: str,
    output: str,
    model_folder: str,
    device: str = 'cpu',
    batch_size: int = 512,
):
    ltp_tokenizer_config = LtpTokenizerConfig(
        model_folder=model_folder,
        device=device,
        batch_size=batch_size,
    )

    # # generate ltp_cache_jsl
    # io.write_json_lines(
    #     'data/temp.jsl',
    #     gen_ltp_tokens(ltp_tokenizer_config, input_csv, token_col),
    #     ensure_ascii=False,
    # )

    # generate extended csv table.
    io.write_csv_lines(
        output,
        gen_ltp_tokens(ltp_tokenizer_config, input_csv, token_col, new_col, True),
        from_dict=True,
    )


if __name__ == '__main__':
    # token_col = '原始名称'
    # new_col = 'query_tokens'
    # texts_csv = r'C:\Users\zhen\Desktop\WorkFiles\search\huantong_proj-data\原始终端拆词-美能华(10000).csv'
    # ltp_model_folder = r'C:\Users\zhen\Desktop\WorkFiles\library_for_search\textdog-data\token\ltp\base'  # noqa
    token_col = 'query_name'
    new_col = 'query_name_tokens'
    RESOURCE = '/home/mnvai/projects/huantong/learning'
    texts_csv = f'{RESOURCE}/data/db-csv/20210615/orgcodecollate.csv'
    ltp_model_folder = f'{RESOURCE}/release/textdog_data/token/ltp/base'
    output_csv = 'data/padded_orgcodecollate.csv'
    debug_tokenize(texts_csv, output_csv, ltp_model_folder, 'cuda')
    # mv data/padded_orgcodecollate.csv /home/mnvai/projects/huantong/learning/release/db-csv/20210615/orgcodecollate.csv  # noqa
