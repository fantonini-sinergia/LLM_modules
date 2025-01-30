import os
import tempfile
import chatbot_constants as k
from LLM_inference.llm import Llm
from vector_databases.embedding import Embedding
from vector_databases.file_processing import extract_page
from vector_databases.vdbs import Vdbs
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


llm_name = os.path.join(k.models_path, k.llm_model)
tokenizer_name = os.path.join(k.models_path, k.llm_tokenizer)
embedding_model_name = os.path.join(k.models_path, k.embedding_model)

context_char_len = len(k.system[0]["content"])
max_content_char_len = k.max_context_len*k.chars_per_token
perm_context_word_len = k.rag_context_word_len*k.perm_context_ratio
temp_context_word_len = k.rag_context_word_len - perm_context_word_len


# LLM model initialization
llm_model = Llm(llm_name, tokenizer_name, bnb_config = k.bnb_config)
print("LLM initialized")

# Embedding model initialization
embedding_model = Embedding(embedding_model_name, k.device)
print("Embedding model initialized")

# permanent vdbs loading and initialization
perm_vdbs = Vdbs.from_dir(
    k.perm_vdbs_folder,
    embedding_model.get_embeddings_for_vdb,
    **k.extend_params,
    )
print("permanent vdbs loaded")

# chat initialization
chat = []
system = k.system

# Endpoint per inferenza
@app.route('/api/infer', methods=['POST'])
def infer():
    global context_char_len
    global chat
    try:
        # Recupera i dati dal corpo della richiesta
        prompt = request.form.get('prompt')

        if not prompt:
            return jsonify({'error': 'Il campo "prompt" è richiesto.'}), 400

        # Get attachments
        attachments = request.files.getlist("files")
        if attachments:
            files = []
            for file in attachments:
                temp_file_path = os.path.join(tempfile.gettempdir(), file.filename)
                file.save(temp_file_path)
                files.append({"name": file.filename, "path": temp_file_path})
            print(f'{len(files)} files attached')
            temp_vdbs = Vdbs.from_files_list(
                files, 
                embedding_model.get_embeddings_for_vdb, 
                False,
                chars_per_word = k.chars_per_word,
                vdbs_params = k.vdbs_params,
                **k.extend_params,
                )
            print("Temporary vdbs created")

            # Get the samples from the temporary vdbs
            samples_from_temp = temp_vdbs.get_rag_samples(
                prompt, 
                embedding_model.get_embeddings_for_question, 
                temp_context_word_len,
                )
            print("retrieved from temporary vdbs")
            
        # Get the samples from the permanent vdbs
        samples_from_perm = perm_vdbs.get_rag_samples(
            prompt, 
            embedding_model.get_embeddings_for_question, 
            perm_context_word_len,
            )
        print("retrieved from permanent vdbs")

        keys = samples_from_perm[0].keys()
        samples = {key: [] for key in keys}
        for d in samples_from_perm:
            for key in keys:
                samples[key] += d[key]
                
        if attachments:
            # Join perm and temp samples
            for d in samples_from_temp:
                for key in keys:
                    samples[key] += d[key]
            print("joined samples from temporary and permanent vdbs")


        # Join sample contents into rag_context and append to the chat
        rag_context = "Usa le seguenti informazioni per rispondere alla domanda.\
            \n\n\nContesto:\n" + \
            "".join(samples["content"]) + \
            "\n\n\nDomanda: "
        print("rag context created")

        # context len adaptation
        context_char_len += (len(rag_context) + len(prompt))
        while context_char_len > max_content_char_len:
            if len(chat)<1:
                raise ValueError(f"context len is {context_char_len} characters, greater than max context len, that is {max_content_char_len} characters")
            context_char_len -= len(chat[0]["content"])
            context_char_len -= len(chat[1]["content"])
            del chat[0:2]
        print(f"chat and context length adapted to be less or equal then {max_content_char_len}")       

        # Generate the answer
        answer = llm_model.llm_qa(
            system + chat + [{"role": "user", "content": rag_context + prompt}],  
            train = False,
            max_new_tokens = k.max_new_tokens,
            temperature = k.temperature,
            top_p = k.top_p,
            )
        print("answer generated")

        # # test answer
        # answer = "This is a test answer"


        # Create a temporary directory for PDFs
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Temporary directory created: {temp_dir}")
            text_sources = []
            pdf_sources = []
            for i, page in enumerate(samples["page"]):
                file_name = samples["file_name"][i]
                file_extension = samples["file_extension"][i]
                if file_extension.upper() == "PDF":
                    file_path = samples["file_path"][i]
                    # the extracted page is page-1, because the func extract_page strarts from 0
                    temp_pdf = extract_page(file_path, page-1, temp_dir)
                    pdf_source = temp_pdf
                    text_source = f"**{file_name}, pagina {page}**"
                    pdf_sources.append(pdf_source)
                else:
                    content = samples["content"][i]
                    text_source = f"**{file_name}, pagina {page}**\n{content}".replace("\n\n", "\n")
                text_sources.append(text_source)
            print(f"rag sources formatted")



        # Update the chat and the context len
        chat = chat + \
        [{"role": "user", "content": prompt}] + \
        [{"role": "assistant", "content": answer}]
        context_char_len = context_char_len + len(answer)
        print("chat and context len updated with new question and answer")


        response = {
            'prompt': prompt,
            'response': answer,
            'text_sources': text_sources,
            'pdf_sources': pdf_sources
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)