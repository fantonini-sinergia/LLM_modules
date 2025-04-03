import os
import traceback
import tempfile
import app.chatbot_constants as k
from app.llm import Llm
from app.vector_databases.embedding import Embedding
from app.vector_databases.file_processing import extract_page
from app.vector_databases.vdbs import Vdbs
from flask import Blueprint, request, jsonify
from fastapi import Body, File, UploadFile

api_chatbot_bp = Blueprint('api_chatbot', __name__)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# LLM model initialization
llm_model = Llm(
    k.system,
    llm = k.llm_model, 
    tokenizer = k.llm_tokenizer, 
    bnb_config = k.bnb_config,
    api_base_url=k.api_base_url,
    api_key=k.api_key,
    )
print("LLM initialized")

# embedder initialization
embedding_model = Embedding(k.embedder, k.device)
print("Embedding model initialized")

# chat initialization
user_chats = {}
system = k.system

# Endpoint per inferenza
@api_chatbot_bp.route('/infer', methods=['POST'])
def infer(
    
):
    try:
        # Recupera i dati dal corpo della richiesta
        # prompt = None,
        data = request.get_json()
        prompt = data.get('prompt', None),
        audio = data.get('audio', None),
        attachments = data.get('files', None)
        chat = data.get('chat', None)
        rag_datasets = data.get('rag_datasets', None)
        # prompt: str = Body(None),
        # audio: UploadFile = File(None),
        # attachments = request.files.getlist("files")
        # chat = Body(None)
        # rag_datasets = File(None)
        """
        search_dataset_url = data.get('search_dataset_url')
        search_dataset_vect_columns = data.get('search_dataset_vect_columns')
        search_only = data.get('search_only')
        """

        # if not existing, initialize the chat
        if chat == None:
            chat = {
                "chat": system,
                "tokens_per_msg": [],
            }

        # Get RAG dataset
        if rag_datasets == None:
            rag_datasets = []
            perm_vdbs = None
        for rag_dataset in rag_datasets:

            """
            Da completare
            """
            rag_dataset_url = rag_datasets
            perm_vdbs = Vdbs.from_dir(
                rag_dataset_url,
                embedding_model.get_embeddings_for_vdb,
                **k.extend_params,
                )
            print("rag_datasets loaded")


        # Get the attachments
        if attachments:
            """
            Da completare
            """
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
                vdbs_params = k.vdbs_params,
                **k.extend_params,
                )
            print("Temporary vdbs created from attachments")
        else:
            temp_vdbs = None
            
        
        # Get the samples from the permanent vdbs
        if perm_vdbs:
            samples_from_perm = perm_vdbs.get_rag_samples(
                prompt, 
                embedding_model.get_embeddings_for_question, 
                nr_bunches = 1,
                )
            print("retrieved from permanent vdbs")

        # Get the samples from the temporary vdbs
        if temp_vdbs:
            samples_from_temp = temp_vdbs.get_rag_samples(
                prompt, 
                embedding_model.get_embeddings_for_question, 
                nr_bunches = 1,
                )
            print("retrieved from temporary vdbs")


        """
        Carloni:
        if search_only:
            # Generate the answer
            import json
            answer = "Ecco che cosa ho trovato per te:" + "\n".join(["\n".join([f"{k}: {v}" for k, v in samp.items()]) for samp in samples_for_search[0]])
            user_chats[user_id]['chat'].append({"question": prompt, "answer": answer})
            print("answer generated")
        """

        samples = None

        if perm_vdbs:
            keys = samples_from_perm[0].keys()
            samples = {key: [] for key in keys}
            for d in samples_from_perm:
                for key in keys:
                    samples[key] += d[key]
            print("joined samples from permanent vdbs")
            if temp_vdbs:
                for d in samples_from_temp:
                    for key in keys:
                        samples[key] += d[key]
                print("joined samples from temporary and permanent vdbs")
        else:
            if temp_vdbs:
                keys = samples_from_temp[0].keys()
                samples = {key: [] for key in keys}
                for d in samples_from_temp:
                    for key in keys:
                        samples[key] += d[key]
                print("joined samples from temporary vdbs")
                

        if samples:
            # Join sample contents into rag_context and append to the chat
            rag_context = "Usa le seguenti informazioni per rispondere alla domanda.\
                \n\n\nContesto:\n" + \
                "".join(samples["content"]) + \
                "\n\n\nDomanda: "
            print("rag context created")

            # Generate the answer
            answer, chat = llm_model.llm_qa(
                chat,
                rag_context + prompt,  
                )
            print("answer generated")
            """"
            temporary single page pdfs creation
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
            """
            text_sources = []
            for i, page in enumerate(samples["page"]):
                file_name = samples["file_name"][i]
                content = samples["content"][i]
                text_source = f"**{file_name}, pagina {page}**\n{content}".replace("\n\n", "\n")
            text_sources.append(text_source)
            print(f"rag sources formatted")

            response = {
                'prompt': prompt,
                'response': answer,
                'text_sources': text_sources
                # 'pdf_sources': pdf_sources
            }

        else:
            # Generate the answer
            answer, chat = llm_model.llm_qa(
                chat,
                prompt,  
                )
            print("answer generated")

            response = {
                'prompt': prompt,
                'response': answer
                # 'text_sources': text_sources,
                # 'pdf_sources': pdf_sources
            }

        return jsonify(response), 200

    except Exception as e:
        error_trace = traceback.format_exc()
        return jsonify({'error': str(e), 'traceback': error_trace}), 500