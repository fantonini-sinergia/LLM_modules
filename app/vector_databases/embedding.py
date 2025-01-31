import torch
from transformers import AutoModel, AutoTokenizer


class Embedding:
    def __init__(self, embedding_model_name, device):
        self.model = AutoModel.from_pretrained(
            embedding_model_name,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            embedding_model_name
        )
        self.device = torch.device(device)

    def get_embeddings_for_vdb(self, text_list):

        encoded_input = self.tokenizer(
                            text_list,
                            padding=True,
                            truncation=True,
                            return_tensors="pt"
        )
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        model_output = self.model(**encoded_input)

        return model_output.last_hidden_state[:, 0].detach().cpu().numpy()[0]
    
    def get_embeddings_for_question(self, text_list):

        encoded_input = self.tokenizer(
                            text_list,
                            padding=True,
                            truncation=True,
                            return_tensors="pt"
        )
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        model_output = self.model(**encoded_input)

        return model_output.last_hidden_state[:, 0].detach().cpu().numpy()