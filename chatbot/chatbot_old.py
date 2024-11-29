import yaml
import os
import torch
import datasets
import chainlit as cl
from llm import Llm
from embedding import Embedding
from file_reading import extract_page
from vdbs_creation import create_dbs, dbs_to_vdbs
from rag_answer_creation import get_rag_samples

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

yaml_file = 'chatbot_config.yaml'
with open(yaml_file, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

device = torch.device(config["device"])
system = config["system"]
llms_path = config["llms_path"]
max_context_len = config["max_context_len"]
bnb_config = config["bnb_config"]
max_new_tokens = config["max_new_tokens"]
temperature = config["temperature"]
top_p = config["top_p"]
embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
bunches_params = config["bunches_params"]
fixed_rag_data = config["fixed_rag_data"]
chat_llm = os.path.join(llms_path, config["chat_llm"]["model"])
chat_tokenizer = os.path.join(llms_path, config["chat_llm"]["tokenizer"])
rag_llm = os.path.join(llms_path, config["rag_llm"]["model"])
rag_tokenizer = os.path.join(llms_path, config["rag_llm"]["tokenizer"])
print(rag_llm)


@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="Chat",
            markdown_description="Chatta con il chatbot specializzato",
            icon="https://picsum.photos/200",
        ),
        cl.ChatProfile(
            name="RAG",
            markdown_description="Fai domande al chatbot sulla documentazione",
            icon="https://picsum.photos/250",
        ),
        cl.ChatProfile(
            name="RAG fissa",
            markdown_description="Fai domande al chatbot sulla documentazione",
            icon="https://picsum.photos/300",
        ),
    ]


@cl.on_chat_start
async def on_chat_start():

    chat_profile = cl.user_session.get("chat_profile")

    # Initialize the chat with the system command
    cl.user_session.set("chat", system)

    if "RAG" in chat_profile:
        llm = rag_llm
        tokenizer = rag_tokenizer
        msg = cl.Message(content='')

        if chat_profile == "RAG":
            # Ask files to the user (Profile "RAG")
            chainlit_format_files = None
            while chainlit_format_files == None:
                chainlit_format_files = await cl.AskFileMessage(
                    content="Carica uno o più files",
                    accept=
                    [
                        "application/msword",
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        "application/pdf",
                        "application/vnd.ms-powerpoint",
                        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        "text/plain",
                        "application/vnd.ms-excel",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    ],
                    max_size_mb = 100,
                    max_files = 10,
                ).send()
            files = [{"name": cff.name, "path": cff.path} for cff in chainlit_format_files]

            # Create the databases (Profile "RAG")
            dbs = create_dbs(files, **bunches_params)

        else:
            # Create the databases (Profile "RAG fissa")
            print("\n\n", "-"*20)
            print("Loading the databases...")
            vdbs = []
            for dir in os.listdir(fixed_rag_data):
                if "database" in dir:
                    vdbs.append(datasets.Dataset.load_from_disk(os.path.join(fixed_rag_data, dir)))
            print(f'Loaded {len(vdbs)} Databases. Number of bunches per db:')
            for vdb in vdbs:
                print(f'- {len(vdb["page"])}')
            print("-"*20, "\n\n")
        
    else:
        llm = chat_llm
        tokenizer = chat_tokenizer

    # LLM model initialization
    llm_model = Llm(bnb_config, llm, tokenizer)
    cl.user_session.set("llm_model", llm_model)

    if "RAG" in chat_profile:
        
        # Embedding model initialization (all "RAG" profiles)
        embedding_model = Embedding(embedding_model_name)
        cl.user_session.set("embedding_model", embedding_model)

        if chat_profile == "RAG":
            #  Vectorization of the databases (Profile "RAG")
            vdbs = dbs_to_vdbs(dbs, embedding_model, device)
        else:
            #  Vectorization of the databases (Profile "RAG fissa")
            for i, vdb in enumerate(vdbs):
                vdb.load_faiss_index('embeddings', f'{fixed_rag_data}\\faiss_{i}.faiss')
            
        cl.user_session.set("vdbs", vdbs)
        msg.content = "Ho analizzato i file, chiedi pure"
        await msg.update()


@cl.on_message
async def on_message(message: cl.Message):

    chat_profile = cl.user_session.get("chat_profile")
    chat = cl.user_session.get("chat")

    if chat_profile == "Chat":

        # Append question to the chat (profile "Chat")
        chat.append({"role": "user", "content": message.content})

    else:
        
        # Get the samples (all "RAG" profiles)
        vdbs = cl.user_session.get("vdbs")
        embedding_model = cl.user_session.get("embedding_model")
        samples = get_rag_samples(
            message, 
            vdbs, 
            embedding_model, 
            device, 
            bunches_params
            )

        # Join sample contents and attach to the text chat (all "RAG" profiles)
        context = "".join(samples["content"])
        chat.append({"role": "user", "content": "Usa le seguenti informazioni per rispondere alla domanda.\n\n\n\
            Contesto:\n"+context+"\n\n\nDomanda:"+message.content})
        

    # Generate the answer
    llm_model = cl.user_session.get("llm_model")
    answer = llm_model.llm_qa(
        chat,  
        train = False,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        )

    if chat_profile == "Chat":

        # stream the answer (Profile: "Chat")
        msg = cl.Message('')
        for i in answer:
            await msg.stream_token(i)
        chat.append({"role": "assistant", "content": answer})

    else:

        # create pdf elements or text elements to view the sources (all "RAG" profiles)
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

        # stream the answer (Profile "Chat")
        for i in answer:
            await msg.stream_token(i)

        # print the sources (all "RAG" profiles)
        await msg.send()
        
        # delete all temporary pdfs (all "RAG" profiles)
        for pdf in temp_pdfs:
            os.remove(pdf)

        # delete last printed answer from chat (all "RAG" profiles)
        chat.pop()

    # Update chat
    cl.user_session.set("chat", chat)