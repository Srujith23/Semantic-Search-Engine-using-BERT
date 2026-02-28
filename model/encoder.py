import torch
from transformers import BertTokenizerFast
from transformers import BertModel
from config import BATCH_SIZE, MODEL_NAME, DEVICE


class BertSentenceEncoder():
    def __init__(self):
        self.model = BertModel.from_pretrained(MODEL_NAME)
        self.tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
        
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.to(DEVICE)
        self.model.eval()
    
    def encode(self, texts):
        all_mask = []
        all_embeddings = []
        for i in range(0,len(texts), BATCH_SIZE):
            batch_texts = texts[i: i+BATCH_SIZE]

            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].to(DEVICE)
            attention_mask = encoded['attention_mask'].to(DEVICE)

            with torch.no_grad():
                outputs = self.model(
                    input_ids = input_ids,
                    attention_mask = attention_mask
                )
            all_mask.append(attention_mask.cpu())
            all_embeddings.append(outputs.last_hidden_state.cpu())
        return torch.vstack(all_embeddings), torch.vstack(all_mask)
