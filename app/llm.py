from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

class Llm:
    def __init__(
            self,
            llm, 
            tokenizer,
            system,
            bnb_config=None, 
            ):
        if bnb_config is not None:
            bnb_config = BitsAndBytesConfig(**bnb_config)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.system = system
        self.system_len = self.tokenizer.apply_chat_template(
            system,
            add_generation_prompt = True,
            return_tensors="pt"
        ).to(self.model.device).shape[-1]
        self.max_input_length = self.tokenizer.model_max_length


    def llm_qa(
            self,
            chat,
            question,
            max_new_tokens=None,  # Allow max_new_tokens to be None
            temperature=0.7, 
            top_p=0.9,
            ):
        print("\n\n", "-"*20)
        print("generating the answer...")

        # chat adaptation
        question = [
            {"role": "user", "content": question}
        ]
        question_len = self.tokenize(question).shape[-1]
        chat["tokens_per_msg"].append(question_len)
        chat["chat"] += question
        msgs_counter = len(chat["tokens_per_msg"])
        while sum(chat["tokens_per_msg"][-msgs_counter:]) > self.max_input_length - self.system_len:
            if msgs_counter<3:
                raise ValueError(f"too long message")
            msgs_counter -= 2
        print(f"chat adapted") 
        
        # tokenize the chat
        input_ids = self.tokenize(self.system + chat["chat"][-msgs_counter:])

        # If max_new_tokens is None, set it to the remaining capacity of the model
        if max_new_tokens is None:
            max_new_tokens = self.max_input_length - input_ids.shape[-1]

        # Ensure the total length does not exceed the model's maximum sequence length
        max_new_tokens = min(max_new_tokens, self.max_input_length - input_ids.shape[-1])

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            max_length=self.max_input_length,  # Ensure total length is within limits
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        
        response = outputs[0][input_ids.shape[-1]:]
        result = self.tokenizer.decode(response, skip_special_tokens=True)
        chat["tokens_per_msg"].append(len(response))
        chat["chat"] += [
            {"role": "assistant", "content": result}
        ]

        print("answer generated")
        print("-"*20, "\n\n")
        return result, chat
    

    def get_max_input_length(self):
        return self.tokenizer.model_max_length
    
    def tokenize(self, chat):
        input_ids = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt= True,
            return_tensors="pt"
        ).to(self.model.device)
        return input_ids

    
    