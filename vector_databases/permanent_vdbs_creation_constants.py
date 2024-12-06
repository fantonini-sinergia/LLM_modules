# source_files_path = r'C:\Users\Sinergia EPC\IDEALE\ontologia'
# save_dir = r'C:\Users\Sinergia EPC\LLM_modules\vector_databases\permanent_vdbs\IDEALE\test_ontologia'
source_files_path = r'C:\Users\Sinergia EPC\IDEALE\ontologi'
save_dir = r'C:\Users\Sinergia EPC\LLM_modules\vector_databases\permanent_vdbs\IDEALE\test_ontologi'

embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
device = "cuda"

as_excel = False
# only for as_excel = True
vect_columns = []
# only for as_excel = False
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
extend_params = {
    "add_words": 300,
    "add_words_nr_word_thr": 150
}