# embedding model constants
models_path = r'C:\Users\FilippoAntonini\OneDrive - Sinergia\models_and_datasets'
embedding_model_name = r'sentence-transformers\all-MiniLM-L6-v2'
# embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
device = "cuda"

# permanent vdbs constants
perm_vdbs_folder = r'C:\Users\FilippoAntonini\OneDrive - Sinergia\LLM_modules\vector_databases\permanent_vdbs\IDEALE\test_ontologia'

# temporary vdbs constants
chars_per_word = 5
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

# extension of the sample
extend_params = {
    "add_words": 300,
    "add_words_nr_word_thr": 150
}