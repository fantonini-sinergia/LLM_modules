import os
import chainlit as cl
import tempfile
import chatbot_constants as k
from LLM_inference.llm import Llm
from vector_databases.embedding import Embedding
from vector_databases.file_processing import extract_page
from vector_databases.vdbs import Vdbs

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


llm_name = os.path.join(k.models_path, k.llm_model)
tokenizer_name = os.path.join(k.models_path, k.llm_tokenizer)
# embedding_model_name = os.path.join(constants.models_path, constants.embedding_model)
embedding_model_name = k.embedding_model

perm_context_word_len = k.rag_context_word_len*k.perm_context_ratio
temp_context_word_len = k.rag_context_word_len - perm_context_word_len



@cl.on_chat_start
async def on_chat_start():

    # Initialize system command, chat and context_len
    cl.user_session.set("system", k.system)
    cl.user_session.set("chat", [])
    cl.user_session.set("context_len", len(k.system[0]["content"]))

    # LLM model initialization
    llm_model = Llm(k.bnb_config, llm_name, tokenizer_name)
    print("LLM initialized")
    cl.user_session.set("llm_model", llm_model)

    # Embedding model initialization
    embedding_model = Embedding(embedding_model_name, k.device)
    print("Embedding model initialized")
    cl.user_session.set("embedding_model", embedding_model)


    # temporary vdbs initialization
    cl.user_session.set("temp_vdbs", [])


@cl.on_message
async def on_message(message: cl.Message):

    system = cl.user_session.get("system")
    chat = cl.user_session.get("chat")
    context_len = cl.user_session.get("context_len")
    llm_model = cl.user_session.get("llm_model")
    embedding_model = cl.user_session.get("embedding_model")
    temp_vdbs = cl.user_session.get("temp_vdbs")

    # Get attachments
    attachments = False
    if not message.elements:
        await cl.Message(content="No file attached").send()
    else:
        attachments = True
        chainlit_format_files = [file for file in message.elements]
        files = [{"name": cff.name, "path": cff.path} for cff in chainlit_format_files]
        print(f'{len(files)} files attached')
        temp_vdbs = Vdbs.from_files_list(
            files, 
            embedding_model.get_embeddings_for_vdb, 
            k.chars_per_word,
            k.vdbs_params,
            )
        print("Temporary vdbs created")
        cl.user_session.set("temp_vdbs", temp_vdbs)

        # Get the samples from the temporary vdbs
        samples_from_temp = samples_from_temp = temp_vdbs[0].get_nearest_examples(
                "embeddings", 
                embededded_question, 
                k = 3
                )
        print("retrieved from temporary vdbs")
        
    if attachments:
        # Join perm and temp samples
        keys = samples_from_temp[0].keys()
        samples = {key: [] for key in keys}
        for d in samples_from_temp:
            for key in keys:
                samples[key] += d[key]
        print("joined samples from temporary vdbs")


    # Join sample contents into rag_context and append to the chat
    rag_context = "Usa le seguenti informazioni per rispondere alla domanda.\
        \n\n\nContesto:\n" + \
        "".join(samples["content"]) + \
        "\n\n\nDomanda: "
    print("rag context created")
    
    # context len adaptation
    context_len += (len(rag_context) + len(message.content))
    while context_len > k.max_context_len*k.chars_per_token:
        context_len -= len(chat[0]["content"])
        context_len -= len(chat[1]["content"])
        try:
            del chat[0:2]
        except:
            print("too long question or too much samples from docs")
    print(f"chat and context length adapted to be less or equal then {k.max_context_len}")       

    # Generate the answer
    answer = llm_model.llm_qa(
        system + chat + [{"role": "user", "content": rag_context + message.content}],  
        train = False,
        max_new_tokens = k.max_new_tokens,
        temperature = k.temperature,
        top_p = k.top_p,
        )
    print("answer generated")

    # Create a temporary directory for PDFs
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Temporary directory created: {temp_dir}")
        sources = []
        for i, page in enumerate(samples["page"]):
            file_name = samples["file_name"][i]
            file_extension = samples["file_extension"][i]
            if file_extension.upper() == "PDF":
                file_path = samples["file_path"][i]
                # the extracted page is page-1, because the func extract_page strarts from 0
                temp_pdf = extract_page(file_path, page-1, temp_dir)
                pdf_source = cl.Pdf(path=temp_pdf)
                text_source = cl.Text(
                    content=f"**{file_name}, pagina {page}**"
                )
                sources.append(pdf_source)
            else:
                content = samples["content"][i]
                text_source = cl.Text(
                    content=f"**{file_name}, pagina {page}**\n{content}".replace("\n\n", "\n")
                ) 
            sources.append(text_source)
        msg = cl.Message('', elements=sources)
        print(f"rag sources formatted")

        # send the answer
        for i in answer:
            await msg.stream_token(i)
        await msg.send()
        print("answer sent to GUI")


    # Update the chat and the context len
    cl.user_session.set("chat", chat + \
    [{"role": "user", "content": message.content}] + \
    [{"role": "assistant", "content": answer}])
    cl.user_session.set("context_len", context_len + len(answer))
    print("chat and context len updated with new question and answer")
    