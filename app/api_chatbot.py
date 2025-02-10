import os
import tempfile
import app.chatbot_constants as k
from app.llm import Llm
from app.vector_databases.embedding import Embedding
from app.vector_databases.file_processing import extract_page
from app.vector_databases.vdbs import Vdbs
from flask import Blueprint, request, jsonify

api_chatbot_bp = Blueprint('api_chatbot', __name__)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


llm_name = os.path.join(k.models_path, k.llm_model)
tokenizer_name = os.path.join(k.models_path, k.llm_tokenizer)
embedding_model_name = os.path.join(k.models_path, k.embedding_model)

# LLM model initialization
llm_model = Llm(
    llm_name, 
    tokenizer_name, 
    k.system,
    bnb_config = k.bnb_config,
    )
print("LLM initialized")

# Embedding model initialization
embedding_model = Embedding(embedding_model_name, k.device)
print("Embedding model initialized")

# chat initialization
user_chats = {}
system = k.system

# Endpoint per inferenza
@api_chatbot_bp.route('/infer', methods=['POST'])
def infer():
    try:
        # Recupera i dati dal corpo della richiesta
        data = request.get_json()
        prompt = data.get('prompt')
        attachments = request.files.getlist("files")
        rag_datasets = data.get('rag_datasets')
        search_dataset_url = data.get('search_dataset_url')
        search_dataset_vect_columns = data.get('search_dataset_vect_columns')
        search_only = data.get('search_only')
        user_id = data.get('user_id')

        if not prompt or not user_id:
            return jsonify({'error': 'I campi "prompt" e "user_id" sono richiesti.'}), 400
        if search_dataset_url and not search_dataset_vect_columns:
            return jsonify({'error': 'Il campo "api_dataset_vect_columns" è richiesto se "api_dataset" è presente.'}), 400

        # Initialize chat history for the user if not already present
        if user_id not in user_chats:
            user_chats[user_id] = {
                'chat': [],
                'tokens_per_msg': [],
            }

        # Get RAG dataset
        if rag_datasets == None:
            rag_datasets = []
        for rag_dataset in rag_datasets:

            """
            Questa parte è da completare non appena hai sistemato Carloni.
            Per ora lascia la lista reg_datasets vuota e non toccare il codice
            """
            if type(rag_dataset) == str:
                # RAG from files
                if rag_dataset == "files":
                    if attachments:
                        pass
                    else:
                        return jsonify({'error': 'You set "files" as rag dataset but missed the attachments'}), 400
                # RAG from directory
                else: 
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
            Questa parte è da completare non appena hai sistemato Carloni.
            Per ora lascia non aggiungere file
            """
            files = []
            for file in attachments:
                temp_file_path = os.path.join(tempfile.gettempdir(), file.filename)
                file.save(temp_file_path)
                files.append({"name": file.filename, "path": temp_file_path})
            print(f'{len(files)} files attached')
            files_temp_vdbs = Vdbs.from_files_list(
                files, 
                embedding_model.get_embeddings_for_vdb, 
                False,
                vdbs_params = k.vdbs_params,
                **k.extend_params,
                )
            print("Temporary vdbs created from attachments")
            
        # Get search dataset from the API
        if search_dataset_url:
            json_temp_vdbs = Vdbs.from_api(
                search_dataset_url, 
                embedding_model.get_embeddings_for_vdb, 
                True,
                vect_columns = search_dataset_vect_columns,
            )
            print("Temporary vdbs created from api")
        
        if not search_only:
            # Get the samples from the permanent vdbs
            if perm_vdbs:
                samples_from_perm = perm_vdbs.get_rag_samples(
                    prompt, 
                    embedding_model.get_embeddings_for_question, 
                    nr_bunches = 1,
                    )
                print("retrieved from permanent vdbs")

            # Get the samples from the files temporary vdbs
            if files_temp_vdbs:
                samples_from_temp = files_temp_vdbs.get_rag_samples(
                    prompt, 
                    embedding_model.get_embeddings_for_question, 
                    nr_bunches = 1,
                    )
                print("retrieved from temporary vdbs")

        # Get the samples from the API temporary vdbs
        if search_only:
            samples_for_search = json_temp_vdbs.get_rag_samples(
                prompt, 
                embedding_model.get_embeddings_for_question, 
                nr_bunches = 1,
                )
            print("retrieved from temporary vdbs")



        if search_only:
            # Generate the answer
            import json
            answer = """Ecco che cosa ho trovato per te:
            """ + "\n".join(["\n".join([f"{k}: {v}" for k, v in samp.items()]) for samp in samples_for_search[0]])
            user_chats[user_id]['chat'].append({"question": prompt, "answer": answer})
            print("answer generated")

        else:
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


            # Generate the answer
            answer, user_chats[user_id] = llm_model.llm_qa(
                user_chats[user_id],
                rag_context + prompt,  
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


        response = {
            'prompt': prompt,
            'response': answer,
            # 'text_sources': text_sources,
            # 'pdf_sources': pdf_sources
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500