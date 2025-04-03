# chat constants
system = [
    {
        "role": "system",
        "content": (
            "Sei un assistente disponibile, rispettoso e onesto. "
            "Rispondi sempre nel modo più utile possibile. Le risposte non devono includere "
            "contenuti dannosi, non etici, razzisti, sessisti, tossici, pericolosi o illegali. "
            "Assicurati che le tue risposte siano socialmente imparziali e positive. "
            "Se non conosci la risposta a una domanda, non condividere informazioni false. "
            "Rispondi sempre e solo in italiano."
        ),
    }
]
chars_per_token = 4
rag_context_ratio = 0.2
perm_rag_context_ratio = 0.2

# llm constants (local)
# models_path = r'C:\Users\FilippoAntonini\OneDrive - Sinergia\LLMs-SIN031\models_and_datasets'
# llm_model = "Qwen\\Qwen2.5-0.5B-Instruct"
# llm_tokenizer = "Qwen\\Qwen2.5-0.5B-Instruct"
# bnb_config = {
#     "load_in_4bit": True,
#     "bnb_4bit_compute_dtype": "float16",
#     "bnb_4bit_quant_type": "nf4",
#     "bnb_4bit_use_double_quant": False
# }
# max_new_tokens = 2048
# temperature = 0.6
# top_p = 0.9
# api_base_url = None
# api_key = None

# llm constants (api)
llm_model = "meta-llama/llama-3.3-70b-instruct:free"
llm_tokenizer = None
bnb_config = None
max_new_tokens = 2048
temperature = 0.6
top_p = 0.9
api_base_url = "https://openrouter.ai/api/v1"
api_key = "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"

# embedding constants
embedder = r'C:\Users\FilippoAntonini\OneDrive - Sinergia\LLMs-SIN031\models_and_datasets\sentence-transformers\all-MiniLM-L6-v2'
device = "cuda"

# permanent vdbs constants
perm_vdbs_folder = r'C:\Users\FilippoAntonini\OneDrive - Sinergia\LLM_modules\permanent_vdbs\FAB_M004_manuali'

# temporary vdbs constants
chars_per_word = 4.8
vdbs_params = [
    {
        "chars_per_bunch": 3000,
        "resplits": 2
    },
    {
        "chars_per_bunch": 500,
        "resplits": 2
    }
]

# extension of the sample
extend_params = {
    "add_chars": 2000,
    "add_chars_nr_char_thr": 1500
}