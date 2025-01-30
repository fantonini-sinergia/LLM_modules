from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

class Llm:
    def __init__(
            self,
            llm, 
            tokenizer,
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

    def llm_qa(
            self,
            chat, 
            train, 
            max_new_tokens, 
            temperature, 
            top_p,
            ):
        print("\n\n", "-"*20)
        print("generating the answer...")
        input_ids = self.tokenizer.apply_chat_template(
            chat,
            # truncation=True,
            # padding=True,
            add_generation_prompt= not train,
            return_tensors="pt"
        ).to(self.model.device)
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            # eos_token_id=terminators,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        response = outputs[0][input_ids.shape[-1]:]
        result = self.tokenizer.decode(response, skip_special_tokens=True)

        print("answer generated")
        print("-"*20, "\n\n")
        return result
    

    # def llm_rag_qa(
    #         self,
    #         chat,
    #         question, 
    #         rag_context,
    #         train, 
    #         max_new_tokens, 
    #         temperature, 
    #         top_p, 
    #         ):
    #     print("\n\n", "-"*20)
    #     print("generating the answer...")
    #     input_ids = self.tokenizer.apply_chat_template(
    #         chat,
    #         # truncation=True,
    #         # padding=True,
    #         add_generation_prompt= not train,
    #         return_tensors="pt"
    #     ).to(self.model.device)
    #     outputs = self.model.generate(
    #         input_ids,
    #         max_new_tokens=max_new_tokens,
    #         # eos_token_id=terminators,
    #         pad_token_id=self.tokenizer.pad_token_id,
    #         do_sample=True,
    #         temperature=temperature,
    #         top_p=top_p,
    #     )
    #     response = outputs[0][input_ids.shape[-1]:]
    #     result = self.tokenizer.decode(response, skip_special_tokens=True)

    #     print("answer generated")
    #     print("-"*20, "\n\n")
    #     return result, len(input_ids)
    
    