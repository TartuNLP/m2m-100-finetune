#!/usr/bin/env python

from transformers.models.m2m_100 import M2M100Tokenizer

def _update_tok_lang_dicts(tokenizer):
  tokenizer.id_to_lang_token = dict(list(tokenizer.id_to_lang_token.items()) + list(tokenizer.added_tokens_decoder.items()))
  tokenizer.lang_token_to_id = dict(list(tokenizer.lang_token_to_id.items()) + list(tokenizer.added_tokens_encoder.items()))
  tokenizer.lang_code_to_token = { k.replace("_", ""): k for k in tokenizer.additional_special_tokens }
  tokenizer.lang_code_to_id = { k.replace("_", ""): v for k, v in tokenizer.lang_token_to_id.items() }

def _add_new_tokens(tokenizer, new_token_list):
  base_idx = len(tokenizer)
  
  upd_idx = 0
  for new_tok in new_token_list:
    if new_tok not in tokenizer.encoder:
      tokenizer.encoder[new_tok] = base_idx + upd_idx
      tokenizer.decoder[base_idx + upd_idx] = new_tok
      upd_idx += 1

def get_extended_m2m100_tokenizer(pretrained_model, new_lang_list, new_token_list):
  lang_toks = ["__" + lang + "__" for lang in new_lang_list]
  
  tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", additional_special_tokens = lang_toks)
  
  _update_tok_lang_dicts(tokenizer)
  
  _add_new_tokens(tokenizer, new_token_list)
  
  return tokenizer
