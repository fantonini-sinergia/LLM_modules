import os
import chainlit as cl
import constants
from llm import Llm
from embedding import Embedding
from file_reading import extract_page
from vdbs import Vdbs

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


llm_name = os.path.join(constants.models_path, constants.llm_model)
tokenizer_name = os.path.join(constants.models_path, constants.llm_tokenizer)
# embedding_model_name = os.path.join(constants.models_path, constants.embedding_model)
embedding_model_name = constants.embedding_model

perm_context_word_len = constants.rag_context_word_len*constants.perm_context_ratio
temp_context_word_len = constants.rag_context_word_len - perm_context_word_len



@cl.on_chat_start
async def on_chat_start():

    # Initialize system command, chat and context_len
    cl.user_session.set("system", constants.system)
    cl.user_session.set("chat", [])
    cl.user_session.set("context_len", len(constants.system[0]["content"]))

    # LLM model initialization
    llm_model = Llm(constants.bnb_config, llm_name, tokenizer_name)
    print("LLM initialized")
    cl.user_session.set("llm_model", llm_model)

    # Embedding model initialization
    embedding_model = Embedding(embedding_model_name)
    print("Embedding model initialized")
    cl.user_session.set("embedding_model", embedding_model)

    # permanent vdbs loading and initialization
    perm_vdbs = Vdbs.from_dir(
        constants.fixed_rag_data,
        embedding_model.get_embeddings_for_vdb
        )
    print("permanent vdbs loaded")
    cl.user_session.set("perm_vdbs", perm_vdbs)

    # temporary vdbs initialization
    cl.user_session.set("temp_vdbs", [])


@cl.on_message
async def on_message(message: cl.Message):

    system = cl.user_session.get("system")
    chat = cl.user_session.get("chat")
    context_len = cl.user_session.get("context_len")
    llm_model = cl.user_session.get("llm_model")
    embedding_model = cl.user_session.get("embedding_model")
    perm_vdbs = cl.user_session.get("perm_vdbs")
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
            constants.vdbs_params
            )
        print("Temporary vdbs created")
        cl.user_session.set("temp_vdbs", temp_vdbs)

        # Get the samples from the temporary vdbs
        samples_from_temp = temp_vdbs.get_rag_samples(
            message.content, 
            embedding_model.get_embeddings_for_question, 
            temp_context_word_len,
            **constants.extend_params
            )
        print("retrieved from temporary vdbs")
        
    # Get the samples from the permanent vdbs
    samples_from_perm = perm_vdbs.get_rag_samples(
        message.content, 
        embedding_model.get_embeddings_for_question, 
        perm_context_word_len,
        **constants.extend_params
        )
    print("retrieved from permanent vdbs")
    
    if attachments:
        # Join perm and temp samples
        keys = samples_from_perm[0].keys()
        samples = {key: [] for key in keys}
        for d in samples_from_perm:
            for key in keys:
                samples[key] += d[key]
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
    context_len += (len(rag_context) + len(message.content))
    while context_len > constants.max_context_len*constants.chars_per_token:
        context_len -= len(chat[0]["content"])
        context_len -= len(chat[1]["content"])
        try:
            del chat[0:2]
        except:
            print("too long question or too much samples from docs")
    print(f"chat and context length adapted to be less or equal then {constants.max_context_len}")       

    # Generate the answer
    answer = llm_model.llm_qa(
        system + chat + [{"role": "user", "content": rag_context + message.content}],  
        train = False,
        max_new_tokens = constants.max_new_tokens,
        temperature = constants.temperature,
        top_p = constants.top_p,
        )
    print("answer generated")

    # create pdf elements or text elements to view the sources
    sources = []
    temp_pdfs = []
    for i, page in enumerate(samples["page"]):
        file_name = samples["file_name"][i]
        file_extension = samples["file_extension"][i]
        if file_extension.upper() == "PDF":
            file_path = samples["file_path"][i]
            # the extracted page is page-1, because the func extract_page strarts from 0
            temp_pdf = extract_page(file_path, page-1)
            temp_pdfs.append(temp_pdf)
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
    
    # delete all temporary pdfs
    for pdf in temp_pdfs:
        os.remove(pdf)
    print("removed temporary pdf for source display")
