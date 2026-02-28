import torch
def mean_pooling(token_embeddings, attention_masks):
    attention_masks = attention_masks.unsqueeze(-1)
    real_tokens = token_embeddings * attention_masks
    sum_embeddings = real_tokens.sum(dim=1)
    token_count = attention_masks.sum(dim=1)
    token_count = token_count.clamp(min=1e-9)
    sentence_emebddings = sum_embeddings/token_count
    return sentence_emebddings