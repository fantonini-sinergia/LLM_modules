import os
import chainlit as cl
import ideale_chatbot_constants as k
from vector_databases.embedding import Embedding
from vector_databases.vdbs import Vdbs

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# embedding_model_name = os.path.join(constants.models_path, constants.embedding_model)
embedding_model_name = os.path.join(
    k.models_path, 
    k.embedding_model_name
    )


@cl.on_chat_start
async def on_chat_start():

    # Embedding model initialization
    embedding_model = Embedding(embedding_model_name, k.device)
    print("Embedding model initialized")
    cl.user_session.set("embedding_model", embedding_model)

    # permanent vdbs loading and initialization
    perm_vdbs = Vdbs.from_dir(
        k.perm_vdbs_folder,
        embedding_model.get_embeddings_for_vdb,
        **k.extend_params,
        )
    print("permanent vdbs loaded")
    cl.user_session.set("perm_vdbs", perm_vdbs)

    # temporary vdbs initialization
    cl.user_session.set("temp_vdbs", [])


@cl.on_message
async def on_message(message: cl.Message):

    embedding_model = cl.user_session.get("embedding_model")
    perm_vdbs = cl.user_session.get("perm_vdbs")
    # temp_vdbs = cl.user_session.get("temp_vdbs")

    # Get attachments
    # attachments = False
    # if not message.elements:
    #     await cl.Message(content="No file attached").send()
    # else:
    #     attachments = True
        # chainlit_format_files = [file for file in message.elements]
        # files = [{"name": cff.name, "path": cff.path} for cff in chainlit_format_files]
        # print(f'{len(files)} files attached')
        # temp_vdbs = Vdbs.from_files_list(
        #     files, 
        #     embedding_model.get_embeddings_for_vdb, 
        #     k.chars_per_word,
        #     k.vdbs_params,
        #     )
        # print("Temporary vdbs created")
        # cl.user_session.set("temp_vdbs", temp_vdbs)

        # Get the samples from the temporary vdbs
        # samples_from_temp = temp_vdbs.get_rag_samples(
        #     message.content, 
        #     embedding_model.get_embeddings_for_question, 
        #     )
        # print("retrieved from temporary vdbs")
        
    # Get the samples from the permanent vdbs
    samples_from_perm = perm_vdbs.get_rag_samples(
        message.content, 
        embedding_model.get_embeddings_for_question, 
        )
    print("retrieved from permanent vdbs")
    
    keys = samples_from_perm[0].keys()
    samples = {key: [] for key in keys}
    for d in samples_from_perm:
        for key in keys:
            samples[key] += d[key]

    # if attachments:
    #     # Join perm and temp samples
    #     for d in samples_from_temp:
    #         for key in keys:
    #             samples[key] += d[key]
    #     print("joined samples from temporary and permanent vdbs")      

    # Create and send the answer
    msg = cl.Message(content = f'Ecco cosa ho trovato\
                     \nmacro_category: {samples["macro_category"]}\
                     \ncategory: {samples["category"]}\
                     \ndescription: {samples["description"]}\
                     \nsub_category: {samples["sub_category"]}\
                     \nurl: {samples["url"]}\
                     \ntags: {samples["tags"]}')
    await msg.send()
    print("answer sent to GUI")