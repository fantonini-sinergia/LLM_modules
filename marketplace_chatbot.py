import os
import marketplace_chatbot_constants as k
from vector_databases.embedding import Embedding
from vector_databases.file_processing import extract_page
from vector_databases.vdbs import Vdbs
from flask import Flask, request, jsonify

app = Flask(__name__)

# Embedding model initialization
embedding_model_name = os.path.join(k.models_path, k.embedding_model)
embedding_model = Embedding(embedding_model_name, k.device)
print("model initialized")

# permanent vdbs loading and initialization
perm_vdbs = Vdbs.from_dir(
    k.perm_vdbs_folder,
    embedding_model.get_embeddings_for_vdb,
    **k.extend_params,
    )
print("permanent vdbs loaded")

# Endpoint per inferenza
@app.route('/api/infer', methods=['POST'])
def infer():
    try:
        # Recupera i dati dal corpo della richiesta
        data = request.get_json()
        prompt = data.get('prompt')

        if not prompt:
            return jsonify({'error': 'Il campo "prompt" è richiesto.'}), 400

        # Generate the answer
        answer = llm_model.llm_qa(
            [{"role": "user", "content":prompt}],  
            train = False,
            max_new_tokens = k.max_new_tokens,
            temperature = k.temperature,
            top_p = k.top_p,
            )
        print("answer generated")

        return jsonify({'prompt': prompt, 'response': answer}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
