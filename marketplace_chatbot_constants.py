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
rag_context_word_len = 1800
max_context_len = 2800
perm_context_ratio = 0.2

# embedding model constants
models_path = r'C:\Users\FilippoAntonini\OneDrive - Sinergia\LLMs-SIN031\models_and_datasets'
embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
embedding_model_name = "sentence-transformers\\all-MiniLM-L6-v2" # still not used
device = "cuda"

# permanent vdbs constants
perm_vdbs_folder = r'C:\Users\Sinergia EPC\LLM_modules\vector_databases\permanent_vdbs\FAB_M004_manuali'

# extension of the sample
extend_params = {
    "add_words": 300,
    "add_words_nr_word_thr": 150
}