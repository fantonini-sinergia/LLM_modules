source_files_path = r''

as_excel = True

embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'

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

device = "cuda"