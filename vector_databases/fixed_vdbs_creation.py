import os
import fixed_vdbs_creation_constants as k
from vector_databases.embedding import Embedding
from vector_databases.vdbs import Vdbs

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# url importing
files = []
for root, dirs, files in os.walk(k.source_files_path):
    for file in files:
        percorso_completo = os.path.join(root, file)
        files.append(percorso_completo)

# Embedding model initialization
embedding_model_name = k.embedding_model
embedding_model = Embedding(embedding_model_name, k.device)
print("Embedding model initialized")

# Create vdbs
temp_vdbs = Vdbs.from_files_list(
    files, 
    embedding_model.get_embeddings_for_vdb, 
    k.chars_per_word,
    k.vdbs_params,
    as_excel = k.as_excel
    )
print("vdbs created")