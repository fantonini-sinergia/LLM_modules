import os
import chatbot_constants as k
from LLM_inference.llm import Llm
from flask import Flask, request, jsonify

app = Flask(__name__)

llm_name = os.path.join(k.models_path, k.llm_model)
tokenizer_name = os.path.join(k.models_path, k.llm_tokenizer)

# LLM model initialization
llm_model = Llm(k.bnb_config, llm_name, tokenizer_name)
print("LLM initialized")

# Endpoint per inferenza
@app.route('/api/infer', methods=['POST'])
def infer():
    try:
        # Recupera i dati dal corpo della richiesta
        data = request.get_json()
        print(data)
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
