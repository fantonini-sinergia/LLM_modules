import os
from pathlib import Path
import json
import permanent_vdbs_creation_constants as k
from app.vector_databases.embedding import Embedding
from app.vector_databases.vdbs import Vdbs

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
    k.as_excel,
    chars_per_word = k.chars_per_word,
    vdbs_params = k.vdbs_params,
    vect_columns = k.vect_columns,
    )
print(vdbs[0].column_names)
print("vdbs created")

# Save on disk
save_dir = Path(k.save_dir)
save_dir.mkdir(parents=True, exist_ok=True)
if k.as_excel:
    for i, vdb in enumerate(vdbs):
        vectorized_columns = vdb.list_indexes()
        for vect_col in vectorized_columns:
            faiss_file = save_dir / f'{i}_{vect_col}.faiss'
            vdb.save_faiss_index(f"{vect_col}", faiss_file)
            vdb.drop_index(f"{vect_col}")
        vdb_file = f'vdb_{i}.hf'
        vdb_path = os.path.join(save_dir,vdb_file)
        vdb.save_to_disk(vdb_path)
else:
    for i, vdb in enumerate(vdbs):
        faiss_file = save_dir / f'{i}.faiss'
        vdb.save_faiss_index('embeddings', faiss_file)
        vdb.drop_index('embeddings')
        vdb_file = f'vdb_{i}.hf'
        vdb_path = os.path.join(save_dir,vdb_file)
        vdb.save_to_disk(vdb_path)

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