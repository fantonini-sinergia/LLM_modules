from LLM_inference import Llm

llm_model = r'C:\Users\Sinergia EPC\LLMs\models_and_datasets\\meta-llama\\Meta-Llama-3.1-8B-Instruct'
llm_tokenizer = r'C:\Users\Sinergia EPC\LLMs\models_and_datasets\\meta-llama\\Meta-Llama-3.1-8B-Instruct'
bnb_config = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": False
}

llm = Llm(bnb_config, llm_model, llm_tokenizer)

# input message
with open('prompt.txt', 'r', encoding='utf-8', errors='replace') as file:
    content = file.read()

message = [{"role": "user", "content": content}]


# # context len adaptation
# context_len += (len(rag_context) + len(message.content))
# while context_len > max_context_len*chars_per_token:
#     context_len -= len(chat[0]["content"])
#     context_len -= len(chat[1]["content"])
#     try:
#         del chat[0:2]
#     except:
#         print("too long question or too much samples from docs")
# print(f"chat and context length adapted to be less or equal then {constants.max_context_len}")

response = llm.llm_qa(message, train=False, max_new_tokens=2048, temperature=0.6, top_p=0.9)

print(response)