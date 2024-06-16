
import os, io, uuid
import streamlit as st
import ScrapperH
from openai import OpenAI
from pinecone import Pinecone

from myfunc.asistenti import read_aad_username
from myfunc.embeddings import prepare_embeddings, do_embeddings, rag_tool_answer
from myfunc.mojafunkcija import (
    positive_login,
    show_logo,
    def_chunk,)
from myfunc.prompts import PromptDatabase, ConversationDatabase
from myfunc.retrievers import TextProcessing, PineconeUtility
from myfunc.varvars_dicts import work_vars

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
text_processor = TextProcessing(gpt_client=client)
pinecone_utility = PineconeUtility()

index_name="neo-positive"
host = os.environ.get("PINECONE_HOST")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
version = "24.04.24"

try:
    x = st.session_state.sys_ragbot
except:
    with PromptDatabase() as db:
        prompt_map = db.get_prompts_by_names(["rag_self_querys", "rag_answer_reformat", "sys_ragbot"],[os.getenv("RAG_SELF_QUERY"), os.getenv("RAG_ANSWER_REFORMAT"), os.getenv("SYS_RAGBOT")])
        st.session_state.rag_self_querys = prompt_map.get("rag_self_querys", "You are helpful assistant that always writes in Serbian.")
        st.session_state.rag_answer_reformat = prompt_map.get("rag_answer_reformat", "You are helpful assistant that always writes in Serbian.")
        st.session_state.sys_ragbot = prompt_map.get("sys_ragbot", "You are helpful assistant that always writes in Serbian.")
   
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = f"RAG Test Bot"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

st.subheader("ChatGPT - auto tools, SQL")

with st.expander("Uputstvo"):
    st.caption("""
    ### Upotreba razlicitih retrievera i prompt augmentation
               
    1. Hybrid search sa podesavanjem aplha i score
    2. SelfQuery with metadata
    3. SQL search
    4. Parent retriever trazi opsirniji doc           
    5. Parent retriever expanduje retrieved data uzimajuci prethodne i naredne chunkove istog doc
    6. Graph pretrazuje Knowledge Graph           
    7. Hyde kreira hipoteticke odgovore za poboljsanje prompta          
    8. Multiquery search kreira dodatne promptove
    9. Cohere Reranking reranks documents using Cohere API
    10. Long Context reorder reorders chunks by putting the most relevant in the beginning and end.
    11. Contextual Compression compresses chunks to relevant parts
    12. WebSearch pretrazuje web
    """)

if "rag_tool" not in st.session_state:
    st.session_state.rag_tool = " " 
if "graph_file" not in st.session_state:
    st.session_state.graph_file = " " 
if "alpha" not in st.session_state:
    st.session_state.alpha = 0.5 
if "score" not in st.session_state:
    st.session_state.score = 0.0
if "byte_data" not in st.session_state:
    st.session_state.byte_data = ""

def main():
    if "username" not in st.session_state:
        st.session_state.username = "positive"
    if deployment_environment == "Azure":    
        st.session_state.username = read_aad_username()
    elif deployment_environment == "Windows":
        st.session_state.username = "lokal"
    elif deployment_environment == "Streamlit":
        st.session_state.username = username

    app_name = "AutoRagBot"
    def get_thread_ids():
        with ConversationDatabase() as db:
            return db.list_threads(app_name, st.session_state.username)
    
    with st.sidebar:
        st.info(f"Prijavljeni ste kao: {st.session_state.username}")

    if "client" not in st.session_state:
        st.session_state.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = work_vars["names"]["openai_model"]
    if "azure_filename" not in st.session_state:
        st.session_state.azure_filename = "altass.csv"
    if "messages" not in st.session_state:
        st.session_state.messages = {}
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    with st.sidebar:
        global phglob
        phglob=st.empty()
        show_logo()
        st.caption("29.02.24")
        
        tab1, tab2, tab3 = st.tabs(["Conversation", "Hybrid params", "Load Graph"])

        with tab3:
            uploaded_file = st.file_uploader("Ucitajte Graph file", key="upl_graph", type="gml")
            if uploaded_file:
                st.session_state.graph_file = uploaded_file.name
                with io.open(st.session_state.graph_file, "wb") as file:
                    file.write(uploaded_file.getbuffer())
            else:
                st.error("Ucitajte Graph file")

        with tab2:
            st.write("Parametri za Hybrid Search")
            st.session_state.alpha = st.slider("Odaberite odnos KW/Semantic (0-KW): ", 0.0, 1.0, 0.5, 0.1)
            st.session_state.score = st.slider("Odaberite minimalan score slicnosti: ", 0.0, 1.0, 0.0, 0.1)

        with tab1:
            operation = st.selectbox("Choose operation", ["New Conversation", "Load Conversation", "Delete Conversation"])

            if operation == "New Conversation":
                st.caption("Create a New Conversation")
                thread_name_input = st.text_input("Thread Name (optional)")

                if st.button("Create"):
                    new_thread_id = str(uuid.uuid4())
                    thread_name = thread_name_input if thread_name_input else f"Thread_{new_thread_id}"

                    conversation_data = [{'role': 'system', 'content': st.session_state.sys_ragbot}]
                    # Check if the thread ID already exists
                    if thread_name not in get_thread_ids():
                        with ConversationDatabase() as db:
                            db.add_sql_record(app_name, st.session_state.username, thread_name, conversation_data)
                        st.success(f"Record {thread_name} added successfully.")
                        st.session_state.thread_id = thread_name
                    else:
                        st.error("Thread ID already exists.")
                            
            elif operation == "Load Conversation":
                    st.caption("Load an Existing Conversation")
                    thread_ids = get_thread_ids()
                    selected_thread_id = st.selectbox("Select Thread ID", thread_ids)

                    if st.button("Select"):
                        with ConversationDatabase() as db:
                            db.query_sql_record(app_name, st.session_state.username, selected_thread_id)
                        st.success(f"Record loaded successfully {selected_thread_id}.")
                        st.session_state.thread_id =  selected_thread_id

            elif operation == "Delete Conversation":
                st.caption("Delete an Existing Conversation")
                thread_ids = get_thread_ids()
                selected_thread_id = st.selectbox("Select Thread ID", thread_ids)

                if st.button("Delete"):
                    try:
                        with ConversationDatabase() as db:
                            db.delete_sql_record(app_name, st.session_state.username, selected_thread_id)
                        st.success(f"Record {selected_thread_id} deleted successfully.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to delete record: {e}")


    with st.sidebar:
        chunk_size, chunk_overlap = def_chunk()
        st.caption("Odaberite operaciju")
        if "podeli_button" not in st.session_state:
            st.session_state["podeli_button"] = False
        if "manage_button" not in st.session_state:
            st.session_state["manage_button"] = False
        if "kreiraj_button" not in st.session_state:
            st.session_state["kreiraj_button"] = False
        if "stats_button" not in st.session_state:
            st.session_state["stats_button"] = False
        if "screp_button" not in st.session_state:
            st.session_state["screp_button"] = False

        if "submit_b" not in st.session_state:
            st.session_state["submit_b"] = False

        if "submit_b2" not in st.session_state:
            st.session_state["submit_b2"] = False

        if "nesto" not in st.session_state:
            st.session_state["nesto"] = 0

        dokum = st.file_uploader(
            "Izaberite dokument/e", key="upload_file", type=["txt", "pdf", "docx", "JSON"])
            
        col1, col2, col3, col4, col5 = st.tabs(["Pripremi Dokument", "Pripremi Websajt", "Kreiraj Knowledge Graph", "Kreiraj Embeding", "Upravljaj sa Pinecone"])

        with col1:
            with st.form(key="podeli", clear_on_submit=False):
                st.session_state.podeli_button = st.form_submit_button(
                    label="Pripremi Dokument",
                    use_container_width=True,
                    help="Podela dokumenta na delove za indeksiranje",
                )
                if st.session_state.podeli_button:
                    st.session_state.nesto = 1
        with col3:
            with st.form(key="graf", clear_on_submit=False):
                st.session_state.podeli_button = st.form_submit_button(
                    label="Kreiraj Knowledge Graph",
                    use_container_width=True,
                    help="Kreiranje Knowledge Graph-a",
                )
                if st.session_state.podeli_button:
                    st.session_state.nesto = 6            
        with col4:
            with st.form(key="kreiraj", clear_on_submit=False):
                st.session_state.kreiraj_button = st.form_submit_button(
                    label="Kreiraj Embeding",
                    use_container_width=True,
                    help="Kreiranje Pinecone Indeksa",
                )
                if st.session_state.kreiraj_button:
                    st.session_state.nesto = 2
        with col5:
        
            with st.form(key="manage", clear_on_submit=False):
                st.session_state.manage_button = st.form_submit_button(
                    label="Upravljaj sa Pinecone",
                    use_container_width=True,
                    help="Manipulacije sa Pinecone Indeksom",
                )
                if st.session_state.manage_button:
                    st.session_state.nesto = 3
                
        with col2:
    
            with st.form(key="screp", clear_on_submit=False):
                st.session_state.screp_button = st.form_submit_button(
                    label="Pripremi Websajt", use_container_width=True, help="Scrape URL"
                )
                if st.session_state.screp_button:
                    st.session_state.nesto = 5
        st.divider()
        phmain = st.empty()

        if st.session_state.nesto == 1:
            with phmain.container():
                if dokum is not None: 
                    prepare_embeddings(chunk_size, chunk_overlap, dokum)
                else:
                    st.error("Uploadujte dokument")
        elif st.session_state.nesto == 2:
            with phmain.container():
                if dokum is not None: 
                    index_name = st.selectbox("Odaberite index", ["neo-positive", "embedings1"], help="Unesite ime indeksa", key="opcije"
                    )
                    if index_name is not None and index_name!=" " and index_name !="" :
                                            
                                if index_name=="embedings1":
            
                                    pinecone=Pinecone(api_key=os.environ.get("PINECONE_API_KEY_STARI"), host="https://embedings1-b1b39e1.svc.us-west1-gcp.pinecone.io") #embedings1 (thai, free)
                                    index = pinecone.Index(host="https://embedings1-b1b39e1.svc.us-west1-gcp.pinecone.io") #embedings1
                                    do_embeddings(dokum, "semantic", os.environ.get("PINECONE_API_KEY_STARI"), host="https://embedings1-b1b39e1.svc.us-west1-gcp.pinecone.io", index_name=index_name, index=index)
                                elif index_name=="neo-positive":
                
                                    pinecone=Pinecone(api_key=os.environ.get("PINECONE_API_KEY_S"), host="https://neo-positive-a9w1e6k.svc.apw5-4e34-81fa.pinecone.io") #neo-positive (thai, serverless, 3072)
                                    index = pinecone.Index(host="https://neo-positive-a9w1e6k.svc.apw5-4e34-81fa.pinecone.io") #neo-positive
                                    do_embeddings(dokum=dokum, tip="hybrid", api_key=os.environ.get("PINECONE_API_KEY_S"), host="https://neo-positive-a9w1e6k.svc.apw5-4e34-81fa.pinecone.io", index_name=index_name, index=index )
                                    
                                else:
                                    st.error("Index ne postoji")
                                    st.stop()
                else:
                    st.error("Uploadujte JSON dokument")
        elif st.session_state.nesto == 3:
            with phmain.container():
                pinecone_utility.obrisi_index()
        
        elif st.session_state.nesto == 5:
            with phmain.container():
                ScrapperH.main(chunk_size, chunk_overlap)
        elif st.session_state.nesto == 6:
            with phmain.container():
                if dokum is not None: 
                    pinecone_utility.create_graph(dokum)
                else:
                    st.error("Uploadujte dokument")


    # Display the existing messages in the current thread
    #  Ensure `messages` and `thread_id` are initialized in session state
    if "messages" not in st.session_state:
        st.session_state.messages = {}
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None
        
    # avatari
    avatar_ai="bot.png" 
    avatar_user = "user.webp"
    avatar_sys= "positivelogo.jpg"
    # Main application logic
    if st.session_state.thread_id is None:
        st.info("Start a conversation by selecting a new or existing conversation.")
    else:
        current_thread_id = st.session_state.thread_id
        # Check if there's an existing conversation in the session state
        if current_thread_id not in st.session_state.messages:
            # If not, initialize it with the conversation from the database or as an empty list
            with ConversationDatabase() as db:
                st.session_state.messages[current_thread_id] = db.query_sql_record(app_name, st.session_state.username, current_thread_id) or []
        if current_thread_id in st.session_state.messages:
                # avatari primena
                for message in st.session_state.messages[current_thread_id]:
                    if message["role"] == "assistant": 
                        with st.chat_message(message["role"], avatar=avatar_ai):
                             st.markdown(message["content"])
                    elif message["role"] == "user":         
                        with st.chat_message(message["role"], avatar=avatar_user):
                             st.markdown(message["content"])
                    else:         
                        with st.chat_message(message["role"], avatar=avatar_sys):
                             st.markdown(message["content"])
           
        # Handle new user input
        if prompt := st.chat_input("What would you like to know?"):

            context = rag_tool_answer(prompt, phglob)
            complete_prompt = st.session_state.rag_answer_reformat.format(prompt=prompt, context=context)
        
            # Append user prompt to the conversation
            st.session_state.messages[current_thread_id].append({"role": "user", "content": complete_prompt})
        
            # Display user prompt in the chat
            with st.chat_message("user", avatar=avatar_user):
                st.markdown(prompt)
                # Assuming `st.session_state.rag_tool` holds some metadata about the tool being used
                st.info(f"Tool in use: {st.session_state.rag_tool}")
                st.caption(f"Prompt with tool result: {complete_prompt}")
        
            # Generate and display the assistant's response
            with st.chat_message("assistant", avatar=avatar_ai):
                message_placeholder = st.empty()
                full_response = ""
                for response in client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=st.session_state.messages[current_thread_id],
                stream=True,
            ):
                    full_response += (response.choices[0].delta.content or "")
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            # Append assistant's response to the conversation
            st.session_state.messages[current_thread_id].append({"role": "assistant", "content": full_response})
        
            # Update the conversation in the database
            with ConversationDatabase() as db:
                db.update_sql_record(app_name, st.session_state.username, current_thread_id, st.session_state.messages[current_thread_id])

       
deployment_environment = os.environ.get("DEPLOYMENT_ENVIRONMENT")

if deployment_environment == "Streamlit":
    name, authentication_status, username = positive_login(main, " ")
else:
    if __name__ == "__main__":
        main()