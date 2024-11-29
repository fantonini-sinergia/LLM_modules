import yaml
import argparse
import os
import torch
from transformers import AutoModel, AutoTokenizer
from vdbs_creation import create_dbs, dbs_to_vdbs

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config")
with open(parser.parse_args().config, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

device = torch.device(config["device"])
embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
bunches_params = config["bunches_params"]
root_directory = config["root_directory"]

# choose the vdb name and create the folder
print("\nWhich name for the Vector Database?\n")
vdb_name = input(">_ ")
save_dir = f'fixed_vector_databases\\{vdb_name}'
os.makedirs(save_dir, exist_ok=True)

# Get all the files in the folder and subfolders
file_paths = []
for root, dirs, files in os.walk(root_directory):
    for file in files:
        file_path = os.path.join(root, file)
        file_paths.append(file_path)
        print(file_path)
files = [{"name": os.path.basename(path), "path": path} for path in file_paths]

# Create the databases (Profile "RAG")
dbs = create_dbs(files, **bunches_params)

# Embedding model initialization (Profile "RAG" or "RAG fissa" or "RAG fissa")
embedding_model = AutoModel.from_pretrained(
    embedding_model_name,
    device_map="auto"
)

# Embedding tokenizer initialization (Profile "RAG" or "RAG fissa" or "RAG fissa")
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)

# Vectorization of the databases (Profile "RAG" or "RAG fissa" or "RAG fissa")
vdbs = dbs_to_vdbs(dbs, embedding_model, embedding_tokenizer, device)

# Save on disk
for i, vdb in enumerate(vdbs):
    vdb.save_faiss_index('embeddings', f'{save_dir}\\faiss_{i}.faiss')
    vdb.drop_index('embeddings')
    vdb.save_to_disk(f'{save_dir}\\database_{i}.hf')