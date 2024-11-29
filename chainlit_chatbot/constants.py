device = "cuda"

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

models_path = r'C:\Users\Sinergia EPC\LLMs\models_and_datasets'

llm_model = "meta-llama\\Meta-Llama-3.1-8B-Instruct"
llm_tokenizer = "meta-llama\\Meta-Llama-3.1-8B-Instruct"

embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'

max_context_len = 2800
rag_context_word_len = 1800
perm_context_ratio = 0.2

bnb_config = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": False
}

max_new_tokens = 2048
temperature = 0.6
top_p = 0.9

embedding_model_name = "sentence-transformers\\all-MiniLM-L6-v2"

fixed_rag_data = r'C:\Users\Sinergia EPC\LLMs\chatbot\fixed_vector_databases\FAB_M004_manuali'

chars_per_word = 5
chars_per_token = 4

vdbs_params = [
    {
        "words_per_bunch": 600,
        "resplits": 2
    },
    {
        "words_per_bunch": 100,
        "resplits": 2
    }
]


extend_params = {
    "add_words": 300,
    "add_words_nr_word_thr": 150
}
