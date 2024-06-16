import streamlit as st
import pandas as pd

import uuid
import os
import json
from io import StringIO

from azure.storage.blob import BlobServiceClient
from openai import OpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper

from pinecone_text.sparse import BM25Encoder
from tqdm.auto import tqdm
from time import sleep
import json
from uuid import uuid4
from io import StringIO
import ScrapperH
from pinecone import Pinecone
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

from myfunc.asistenti import read_aad_username, load_data_from_azure, upload_data_to_azure
from myfunc.mojafunkcija import (
    positive_login,
    def_chunk,
    pinecone_stats,
)

from myfunc.prompts import PromptDatabase
from myfunc.retrievers import (TextProcessing, PineconeUtility, HybridQueryProcessor)
from myfunc.varvars_dicts import work_vars

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
text_processor = TextProcessing(gpt_client=client)
pinecone_utility = PineconeUtility()

index_name="neo-positive"
#api_key = os.environ.get("PINECONE_API_KEY")
host = os.environ.get("PINECONE_HOST")
version = "28.03.24"

try:
    x = st.session_state.sys_ragbot
except:
    with PromptDatabase() as db:
        prompt_map = db.get_prompts_by_names(["format_search_results", "sys_struct_json", "sys_ragbot"],[os.getenv("FORMAT_SEARCH_RESULTS"), os.getenv("SYS_STRUCT_JSON"), os.getenv("SYS_RAGBOT")])
        st.session_state.format_search_results = prompt_map.get("format_search_results", "You are helpful assistant")
        st.session_state.sys_struct_json = prompt_map.get("sys_struct_json", "You are helpful assistant")
        st.session_state.sys_ragbot = prompt_map.get("sys_ragbot", "You are helpful assistant")

global username

def prepare_embeddings(chunk_size, chunk_overlap, dokum):
    skinuto = False
    napisano = False

    file_name = "chunks.json"
    with st.form(key="my_form_prepare", clear_on_submit=False):
        
        # define delimiter
        text_delimiter = st.text_input(
            "Unesite delimiter: ",
            help="Delimiter se koristi za podelu dokumenta na delove za indeksiranje. Prazno za paragraf",
        )
        # define prefix
        text_prefix = st.text_input(
            "Unesite prefiks za tekst: ",
            help="Prefiks se dodaje na početak teksta pre podela na delove za indeksiranje",
        )
        add_schema = st.radio(
            "Da li želite da dodate Metadata (Dodaje ime i temu u metadata): ",
            ("Ne", "Da"),
            key="add_schema_doc",
            help="Dodaje u metadata ime i temu",
        )
        add_pitanje = st.radio(
            "Da li želite da dodate pitanje: ",
            ("Ne", "Da"),
            key="add_pitanje_doc",
            help="Dodaje pitanje u text",
        )
        semantic = st.radio(
            "Da li želite semantic chunking: ",
            ("Ne", "Da"),
            key="semantic",
            help="Greg Kamaradt Semantic Chunker",
        )
        st.session_state.submit_b = st.form_submit_button(
            label="Submit",
            help="Pokreće podelu dokumenta na delove za indeksiranje",
        )
        st.info(f"Chunk veličina: {chunk_size}, chunk preklapanje: {chunk_overlap}")
        if len(text_prefix) > 0:
            text_prefix = text_prefix + " "

        if dokum is not None and st.session_state.submit_b == True:
            
            data=pinecone_utility.read_uploaded_file(dokum, text_delimiter)
            # Split the document into smaller parts, the separator should be the word "Chapter"
            if semantic == "Da":
                text_splitter = SemanticChunker(OpenAIEmbeddings())
            else:
                text_splitter = CharacterTextSplitter(
                        separator=text_delimiter,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )

            texts = text_splitter.split_documents(data)


            # # Create the OpenAI embeddings
            st.success(f"Učitano {len(texts)} tekstova")

            # Define a custom method to convert Document to a JSON-serializable format
            output_json_list = []
            
            # Loop through the Document objects and convert them to JSON
            i = 0
            for document in texts:
                i += 1
                if add_pitanje=="Da":
                    pitanje = text_processor.add_question(document.page_content) + " "
                    st.info(f"Dodajem pitanje u tekst {i}")
                else:
                    pitanje = ""
    
                output_dict = {
                    "id": str(uuid4()),
                    "chunk": i,
                    "text": text_processor.format_output_text(text_prefix, pitanje, document.page_content),
                    "source": document.metadata.get("source", ""),
                    "date": text_processor.get_current_date_formatted(),
                }

                if add_schema == "Da":
                    try:
                        person_name, topic = text_processor.add_self_data(document.page_content)
                    except Exception as e:
                        st.write(f"An error occurred: {e}")
                        person_name, topic = "John Doe", "Any"
    
                    output_dict["person_name"] = person_name
                    output_dict["topic"] = topic
                    st.success(f"Processing {i} of {len(texts)}, {person_name}, {topic}")

                output_json_list.append(output_dict)
                

            # # Specify the file name where you want to save the JSON data
            json_string = (
                "["
                + ",\n".join(
                    json.dumps(d, ensure_ascii=False) for d in output_json_list
                )
                + "]"
            )

            # Now, json_string contains the JSON data as a string

            napisano = st.info(
                "Tekstovi su sačuvani u JSON obliku, downloadujte ih na svoj računar"
            )

    if napisano:
        file_name = os.path.splitext(dokum.name)[0]
        skinuto = st.download_button(
            "Download JSON",
            data=json_string,
            file_name=f"{file_name}.json",
            mime="application/json",
        )
    if skinuto:
        st.success(f"Tekstovi sačuvani na {file_name} su sada spremni za Embeding")


def do_embeddings(dokum, tip, api_key, host, index_name, index):
    with st.form(key="my_form_do", clear_on_submit=False):
        err_log = ""
        # Read the texts from the .txt file
        chunks = []
        
        # Now, you can use stored_texts as your texts
        namespace = st.text_input(
            "Unesi naziv namespace-a: ",
            help="Naziv namespace-a je obavezan za kreiranje Pinecone Indeksa",
        )
        submit_b2 = st.form_submit_button(
            label="Submit", help="Pokreće kreiranje Pinecone Indeksa"
        )
        if submit_b2 and dokum and namespace:
            stringio = StringIO(dokum.getvalue().decode("utf-8"))

            # Directly load the JSON data from file content
            data = json.load(stringio)

            # Initialize lists outside the loop
            my_list = []
            my_meta = []

            # Process each JSON object in the data
            for item in data:
                # Append the text to my_list
                my_list.append(item['text'])
    
                # Append other data to my_meta
                meta_data = {key: value for key, value in item.items() if key != 'text'}
                my_meta.append(meta_data)
                
            if tip == "hybrid":
                embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
                bm25_encoder = BM25Encoder()
                # fit tf-idf values on your corpus
                bm25_encoder.fit(my_list)

                retriever = PineconeHybridSearchRetriever(
                        embeddings=embeddings,
                        sparse_encoder=bm25_encoder,
                        index=index,
                )
                            
                retriever.add_texts(texts=my_list, metadatas=my_meta, namespace=namespace)
                
            else:
                embed_model = "text-embedding-ada-002"
            
                batch_size = 100  # how many embeddings we create and insert at once
                progress_text2 = "Insertovanje u Pinecone je u toku."
                progress_bar2 = st.progress(0.0, text=progress_text2)
                ph2 = st.empty()
            
                for i in tqdm(range(0, len(data), batch_size)):
                    # find end of batch
                    i_end = min(len(data), i + batch_size)
                    meta_batch = data[i:i_end]

                    # get texts to encode
                    ids_batch = [x["id"] for x in meta_batch]
                    texts = [x["text"] for x in meta_batch]
                
                    # create embeddings (try-except added to avoid RateLimitError)
                    try:
                        res = client.embeddings.create(input=texts, model=embed_model)

                    except Exception as e:
                        done = False
                        print(e)
                        while not done:
                            sleep(5)
                            try:
                                res = client.embeddings.create(input=texts, model=embed_model)
                                done = True

                            except:
                                pass

                    embeds = [item.embedding for item in res.data]

                    # Check for [nan] embeddings
            
                    if len(embeds) > 0:
                    
                        to_upsert = list(zip(ids_batch, embeds, meta_batch))
                    else:
                        err_log += f"Greška: {meta_batch}\n"
                    # upsert to Pinecone
                    err_log += f"Upserting {len(to_upsert)} embeddings\n"
                    with open("err_log.txt", "w", encoding="utf-8") as file:
                        file.write(err_log)

                    index.upsert(vectors=to_upsert, namespace=namespace)
                    stodva = len(data)
                    if i_end > i:
                        deo = i_end
                    else:
                        deo = i
                    progress = deo / stodva
                    l = int(deo / stodva * 100)

                    ph2.text(f"Učitano je {deo} od {stodva} linkova što je {l} %")

                    progress_bar2.progress(progress, text=progress_text2)
                    

            # gives stats about index
            st.info("Napunjen Pinecone")

            st.success(f"Sačuvano u Pinecone-u")
            pinecone_stats(index, index_name)


def main():
    if "username" not in st.session_state:
        st.session_state.username = "positive"
    if deployment_environment == "Azure":    
        st.session_state.username = read_aad_username()
    elif deployment_environment == "Windows":
        st.session_state.username = "lokal"
    elif deployment_environment == "Streamlit":
        st.session_state.username = username
    
    with st.sidebar:
        st.info(
            f"Prijavljeni ste kao: {st.session_state.username}")

    if "blob_service_client" not in st.session_state:
        st.session_state.blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZ_BLOB_API_KEY"))
    if "messages" not in st.session_state:
        st.session_state.messages = {}
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = 1
    if "client" not in st.session_state:
        st.session_state.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = work_vars["names"]["openai_model"]
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = st.session_state.sys_ragbot
    if "azure_filename" not in st.session_state:
        st.session_state.azure_filename = "altass.csv"

    st.title("ChatGPT - no assistants, no langchain")


    def ensure_csv_structure(bsc):
        container_name = "positive-user"
        blob_name = "altass.csv"
        container_client = bsc.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_name)
        
        try:
            csv_content = blob_client.download_blob().readall().decode("utf-8")
            if not csv_content:
                raise Exception("CSV file is empty.")
            existing_data = pd.read_csv(StringIO(csv_content))
            required_columns = ['Username', 'Thread ID', 'Thread Name', 'Conversation']
            missing_columns = [col for col in required_columns if col not in existing_data.columns]
            for col in missing_columns:
                existing_data[col] = pd.NA  # Add missing columns with NA values
            
            _ = """
            if missing_columns:
                # If there were missing columns, upload the corrected CSV
                blob_client.upload_blob(existing_data.to_csv(index=False), overwrite=True)
                print("CSV structure was corrected and uploaded.")
            else:
                print("CSV structure is correct.")
            """
        except Exception as e:
            # If the CSV does not exist or is empty, create it with the required structure
            print(f"CSV file does not exist or another error occurred: {e}. Initializing...")
            # Create CSV with 'Username' column if it doesn't exist
            df = pd.DataFrame(columns=required_columns)
            blob_client.upload_blob(df.to_csv(index=False), overwrite=True)
            print("Initialized and uploaded the CSV file to Azure Blob Storage.")

    ensure_csv_structure(st.session_state.blob_service_client)


    def web_search_process(query: str) -> str:
        return GoogleSerperAPIWrapper(environment=os.environ["SERPER_API_KEY"]).run(query)


    def create_structured_prompt(user_query):
        return [
            {"role": "system", "content": st.session_state.sys_struct_json},
            {"role": "user", "content": user_query}
        ]


    def get_structured_decision_from_model(user_query):
        response = st.session_state.client.chat.completions.create(
            model=st.session_state.openai_model,
            messages=create_structured_prompt(user_query),
        )
        print(response)
        # Attempt to parse the last message as structured JSON
        try:
            # Extract the decision text from the response
            decision_text = response.choices[0].message.content
            # Correctly remove markdown code block syntax (triple backticks and "json" keyword) if present
            decision_text_cleaned = decision_text.replace('```json', '').replace('```', '').strip()
            # Attempt to parse the cleaned text as structured JSON
            decision_json = json.loads(decision_text_cleaned)
            return decision_json
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Failed to decode JSON from model response: {e}")
            return None


    def execute_decision_based_on_json(decision_json, user_query):
        if not decision_json:
            return "I'm sorry, I couldn't process the request properly."

        tool = decision_json.get("tool")
        if tool == "HybridQueryProcessor":
            # Assuming you want to use the hybrid_query method directly
            # You might need to adjust the parameters based on your needs
            processor = HybridQueryProcessor()  # Initialize with any needed parameters
            results, tokens = processor.hybrid_query(user_query)  # Directly use the query text
            # Format the results into a response string
            response = format_search_results_from_tools("\n\n".join([f"Source: {result['source']}, Content: {result['page_content']}" for result in results]), user_query)
        elif tool == "WebSearchProcess":
            x = web_search_process(user_query)
            response = format_search_results_from_tools(x, user_query)
        elif tool == "DirectResponse":
            # response = decision_json.get("response", "Here's what I found based on my knowledge.")
            response = st.session_state.client.chat.completions.create(
                model=st.session_state.openai_model,
                messages=[
                    {"role": "system", "content": st.session_state.system_prompt},
                    {"role": "user", "content": user_query}
                ])
            response = response.choices[0].message.content.strip()
        else:
            return "The requested tool is not recognized."
        
        return response


    def format_search_results_from_tools(results, user_query):
        prompt = st.session_state.format_search_results.format(user_query=user_query, results=results)
        
        response = st.session_state.client.chat.completions.create(
            model=st.session_state.openai_model,
            messages=[{"role": "system", "content": st.session_state.system_prompt},
                    {"role": "user", "content": prompt}],
        )
        try:
            synthesized_response = response.choices[0].message.content.strip()
            return synthesized_response
        except AttributeError:
            print("Failed to generate a synthesized response.")
            return "I couldn't find a concise summary based on the search results."
        

    def update_and_upload_thread_data(thread_id, thread_name, bsc):
        existing_data = load_data_from_azure(bsc=bsc, filename=st.session_state.azure_filename, username=st.session_state.username, is_from_altass=True)
        # Include username in the new row to be added
        new_row = pd.DataFrame([[st.session_state.username, thread_id, thread_name, ""]], columns=['Username', 'Thread ID', 'Thread Name', 'Conversation'])

        # Check if the thread ID already exists in the existing data
        if 'Thread ID' in existing_data.columns and not existing_data[existing_data['Thread ID'] == thread_id].empty:
            print("Thread already exists. Skipping update.")
            return
        else:
            if 'Thread ID' not in existing_data.columns:
                print("'Thread ID' column is missing in the DataFrame.")

        # Append the new row to the existing DataFrame
        updated_data = pd.concat([existing_data, new_row], ignore_index=True)
        
        # Upload updated data to Azure
        upload_data_to_azure(updated_data, st.session_state.azure_filename)


    def save_conversation_to_azure(thread_id, messages, bsc):
        thread_name = "Thread_" + str(thread_id)
        conversation = "\n".join([f"{message['role'].title()}: {message['content']}" for message in messages])
        existing_data = load_data_from_azure(bsc=bsc, filename=st.session_state.azure_filename, username=st.session_state.username, is_from_altass=True)
        
        if 'Conversation' not in existing_data.columns:
            existing_data['Conversation'] = pd.NA
        row_index = existing_data[existing_data['Thread ID'] == thread_id].index
        if not row_index.empty:
            existing_data.at[row_index[0], 'Conversation'] = conversation
        else:
            new_row = pd.DataFrame([[st.session_state.username, thread_id, thread_name, conversation]], columns=['Username', 'Thread ID', 'Thread Name', 'Conversation'])
            existing_data = pd.concat([existing_data, new_row], ignore_index=True)
        
        upload_data_to_azure(existing_data, st.session_state.azure_filename)


    def list_saved_threads(bsc):
        existing_data = load_data_from_azure(bsc=bsc, filename=st.session_state.azure_filename, username=st.session_state.username, is_from_altass=True)
        # Check if the DataFrame contains the expected columns
        if 'Thread ID' in existing_data.columns and 'Thread Name' in existing_data.columns:
            return existing_data[['Thread ID', 'Thread Name']].to_dict('records')
        else:
            # Log an error or take corrective action here
            print("Error: CSV file is missing the required 'Thread ID' and 'Thread Name' columns.")
            return []


    def load_thread_from_azure(thread_id, bsc):
        existing_data = load_data_from_azure(bsc=bsc, filename=st.session_state.azure_filename, username=st.session_state.username, is_from_altass=True)
        row = existing_data.loc[existing_data['Thread ID'] == thread_id]
        if not row.empty:
            conversation = row['Conversation'].iloc[0]
            messages = [{"role": line.split(": ")[0].lower(), "content": line.split(": ", 1)[1]} for line in conversation.split("\n") if ": " in line]
            return messages
        else:
            return []


    def delete_thread_from_azure(thread_id, bsc):
        existing_data = load_data_from_azure(bsc=bsc, filename=st.session_state.azure_filename, username=st.session_state.username, is_from_altass=True)
        
        # Check if the thread ID exists and remove it
        if thread_id in existing_data['Thread ID'].values:
            # Remove the selected thread by its ID
            updated_data = existing_data[existing_data['Thread ID'] != thread_id]
            
            # Upload the updated DataFrame back to Azure Blob Storage
            upload_data_to_azure(updated_data, st.session_state.azure_filename)
            print(f"Thread {thread_id} deleted successfully.")
        else:
            print(f"Thread {thread_id} not found.")


    with st.sidebar:
        # Input for new thread name with a clear default instruction
        thread_name_input = st.text_input("Name your thread (leave blank for default naming)", key='thread_name_input')

        if st.button('Start New Conversation'):
            new_thread_id = str(uuid.uuid4())
            # Use the input value if provided, else default to "Thread_" + new_thread_id
            thread_name = thread_name_input if thread_name_input else f"Thread_{new_thread_id}"
            st.session_state["thread_id"] = new_thread_id
            st.session_state.messages[new_thread_id] = []
            update_and_upload_thread_data(new_thread_id, thread_name, st.session_state.blob_service_client)
            st.success(f'New conversation started with Thread ID: {new_thread_id}')

        # Dropdown for selecting an existing thread
        thread_options = list_saved_threads(st.session_state.blob_service_client)
        thread_names = [thread['Thread Name'] for thread in thread_options]
        selected_thread_name = st.selectbox("Select a thread to load", options=[''] + thread_names)
        if selected_thread_name:
            selected_thread_id = next((thread['Thread ID'] for thread in thread_options if thread['Thread Name'] == selected_thread_name), None)
            if selected_thread_id:
                loaded_messages = load_thread_from_azure(selected_thread_id, st.session_state.blob_service_client)
                st.session_state["thread_id"] = selected_thread_id
                st.session_state.messages[selected_thread_id] = loaded_messages

        # Dropdown for selecting a thread to delete
        delete_thread_name = st.sidebar.selectbox("Select a thread to delete", options=[''] + thread_names, key='delete_thread')
        if st.sidebar.button('Delete Selected Thread'):
            if delete_thread_name:
                delete_thread_id = next((thread['Thread ID'] for thread in thread_options if thread['Thread Name'] == delete_thread_name), None)
                if delete_thread_id:
                    delete_thread_from_azure(delete_thread_id, st.session_state.blob_service_client)
                    st.sidebar.success(f'Thread "{delete_thread_name}" deleted')
                    
                    # Check if the deleted thread is the currently opened thread
                    if delete_thread_id == st.session_state["thread_id"]:
                        # Clear the chat display by resetting messages and thread_id
                        st.session_state.messages[delete_thread_id] = []
                        st.session_state["thread_id"] = None  # Indicate no thread is currently selected
                        
                    st.rerun()

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

    current_thread_id = st.session_state["thread_id"]
    with st.container():
        if prompt := st.chat_input("What is up?"):

            st.session_state.messages[current_thread_id].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            decision_json = get_structured_decision_from_model(prompt)
            response = execute_decision_based_on_json(decision_json, prompt)
            # Display the response from either the chosen tool or the direct answer
            with st.chat_message("assistant"):
                st.markdown(response)

            # Append the response to the conversation history and save it
            st.session_state.messages[current_thread_id].append({"role": "assistant", "content": response})
            save_conversation_to_azure(current_thread_id, st.session_state.messages[current_thread_id], st.session_state.blob_service_client)

    if current_thread_id in st.session_state.messages:
        for message in st.session_state.messages[current_thread_id]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


deployment_environment = os.environ.get("DEPLOYMENT_ENVIRONMENT")

if deployment_environment == "Streamlit":
    name, authentication_status, username = positive_login(main, " ")
else:
    if __name__ == "__main__":
        main()