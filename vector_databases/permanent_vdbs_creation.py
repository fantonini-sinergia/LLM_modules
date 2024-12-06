import os
from pathlib import Path
import json
import permanent_vdbs_creation_constants as k
from embedding import Embedding
from vdbs import Vdbs

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# url importing
files = []
for root, dirs, all_files in os.walk(k.source_files_path):
    for file in all_files:
        percorso_completo = os.path.join(root, file)
        files.append({
            "name": file,
            "path": percorso_completo,
            })
        break

# Embedding model initialization
embedding_model_name = k.embedding_model
embedding_model = Embedding(embedding_model_name, k.device)
print("Embedding model initialized")

# Create vdbs
vdbs = Vdbs.from_files_list(
    files, 
    embedding_model.get_embeddings_for_vdb, 
    k.chars_per_word,
    k.vdbs_params,
    as_excel = k.as_excel,
    vect_columns = k.vect_columns,
    )
print("vdbs created")

# Save on disk
save_dir = Path(k.save_dir)
save_dir.mkdir(parents=True, exist_ok=True)
if k.as_excel:
    for i, vdb in enumerate(vdbs):
        for col in k.vect_columns:
            faiss_file = save_dir / f'{i}_{col}_embed.faiss'
            vdb.save_faiss_index(f"{col}_embed", faiss_file)
            vdb.drop_index(f"{col}_embed")
        vdb_file = save_dir / f'vdb_{i}.hf'
        vdb.save_to_disk(vdb_file)
else:
    for i, vdb in enumerate(vdbs):
        faiss_file = save_dir / f'{i}.faiss'
        vdb.save_faiss_index('embeddings', faiss_file)
        vdb.drop_index('embeddings')
        vdb_file = save_dir / f'vdb_{i}.hf'
        vdb.save_to_disk(vdb_file)

parameters = {
    'as_excel': k.as_excel,
    'vect_columns': k.vect_columns,
    'chars_per_word': k.chars_per_word,
    'vdbs_params': k.vdbs_params,
    'extend_params': k.extend_params,
}
parameters_file = save_dir / "parameters.json"
with open(parameters_file, "w") as f:
    json.dump(parameters, f)