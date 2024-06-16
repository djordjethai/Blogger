
import os, uuid
import streamlit as st
from openai import OpenAI
from myfunc.asistenti import read_aad_username
from myfunc.mojafunkcija import (
    positive_login,
    show_logo, initialize_session_state, 
    check_openai_errors, 
    read_txts
)
from myfunc.prompts import PromptDatabase, ConversationDatabase
from myfunc.varvars_dicts import work_vars, work_prompts
import streamlit.components.v1 as components
import html

client = OpenAI()

version = "24.04.24"

mprompts = work_prompts()

default_values = {
    "prozor": st.query_params.get('prozor', "d"),
    "_last_speech_to_text_transcript_id": 0,
    "_last_speech_to_text_transcript": None,
    "success": False,
    "toggle_state": False,
    "button_clicks": False,
    "prompt": '',
    "vrsta": False,
    "messages": {},
    "image_ai": None,
    "thread_id": 'ime',
    "filtered_messages": "",
    "selected_question": None,
    "username": "positive",
    "openai_model": work_vars["names"]["openai_model"],
    "azure_filename": "altass.csv",
    "app_name": "BlogBot",
    "upload_key": 0,
}

initialize_session_state(default_values)
if st.session_state.thread_id not in st.session_state.messages:
    #st.session_state.messages[st.session_state.thread_id] = [{'role': 'system', 'content': mprompts["sys_ragbot"]}]
    st.session_state.messages[st.session_state.thread_id] = [{'role': 'system', 'content': "Ti si expert za pisanje tekstova na srpskom jeziku, narocito u oblasti IT services i digitalne transfomacije. Kreiraj opsirne tekstove iz zadate teme, uvek na srpskom jeziku"}]
if "temp" not in st.session_state:
    st.session_state.temp=0.0
       
initialize_session_state(default_values)

st.subheader("ChatGPT - kreira tekstove")

with st.expander("Uputstvo"):
    st.caption("### Upotreba Kreatora Tekstova \n" 
               " Na osnovu uploadovanih dokumenata zadajte instrukcije za pisanje teksta." 
               " Sa strane mozete podesiti temperaturu u zavisnosti od toga koliko kreativan odgovor zelite."
               " Nakon prve verzije mozete traziti ispravke. Uvek mozete nastaviti istu konverzaciju kasnije." 
               " Mozete sacuvati konverzaciju ili konkretan odgovor."
               " U ovoj verziji Asistent nema pristup ni internetu ni internim podacima kompanije!")
    st.caption(
               " COMING SOON - Mozete koristiti biblioteku instrukcija.")

def copy_to_clipboard(message):
    sanitized_message = html.escape(message)  # Escape the message to handle special HTML characters
    # Create an HTML button with embedded JavaScript for clipboard functionality and custom CSS
    html_content = f"""
    <html>
    <head>
        <style>
            #copyButton {{
                background-color: #454654;  /* Dark gray background */
                color: #f1f1f1;            /* Our white text color */
                border: none;           /* No border */
                border-radius: 8px;     /* Rounded corners */
                cursor: pointer;        /* Pointer cursor on hover */
                outline: none;          /* No focus outline */
                font-size: 20px;        /* Size */
            }}
            #textArea {{
                opacity: 0; 
                position: absolute; 
                pointer-events: none;
            }}
        </style>
    </head>
    <body>
    <textarea id="textArea">{sanitized_message}</textarea>
    <textarea id="textArea" style="opacity: 0; position: absolute; pointer-events: none;">{sanitized_message}</textarea>
    <button id="copyButton" onclick='copyTextToClipboard()'>📄</button>
    <script>
    function copyTextToClipboard() {{
        var textArea = document.getElementById("textArea");
        var copyButton = document.getElementById("copyButton");
        textArea.style.opacity = 1;
        textArea.select();
        try {{
            var successful = document.execCommand('copy');
            var msg = successful ? '✔️' : '❌';
            copyButton.innerText = msg;
        }} catch (err) {{
            copyButton.innerText = 'Failed!';
        }}
        textArea.style.opacity = 0;
        setTimeout(function() {{ copyButton.innerText = "📄"; }}, 3000);  // Change back after 3 seconds
    }}
    </script>
    </body>
    </html>
    """
    components.html(html_content, height=50)

def main():
    if "username" not in st.session_state:
        st.session_state.username = "positive"
    if deployment_environment == "Azure":    
        st.session_state.username = read_aad_username()
    elif deployment_environment == "Windows":
        st.session_state.username = "lokal"
    elif deployment_environment == "Streamlit":
        st.session_state.username = username
    #conversation_data = [{'role': 'system', 'content': mprompts["sys_ragbot"]}]
    conversation_data = [{'role': 'system', 'content': "Ti si expert za pisanje tekstova na srpskom jeziku, narocito u oblasti IT services i digitalne transfomacije. Kreiraj opsirne tekstove iz zadate teme, uvek na srpskom jeziku"}]
    app_name = "TekstKreator"
    def get_thread_ids():
        with ConversationDatabase() as db:
            return db.list_threads(app_name, st.session_state.username)
    
    with st.sidebar:
        st.info(f"Prijavljeni ste kao: {st.session_state.username}")

    if "client" not in st.session_state:
        st.session_state.client = OpenAI()
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = work_vars["names"]["openai_model"]
    if "messages" not in st.session_state:
        st.session_state.messages = {}
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None


    with st.sidebar:
        show_logo()
        st.caption("16.06.24")
        st.session_state.temp=st.slider("Temperatura", 0.0, 1.0, 0.0, 0.1)
        operation = st.selectbox("Choose operation", ["New Conversation", "Load Conversation", "Delete Conversation"])

        if operation == "New Conversation":
            st.caption("Create a New Conversation")
            thread_name_input = st.text_input("Thread Name (optional)")

            if st.button("Create"):
                new_thread_id = str(uuid.uuid4())
                thread_name = thread_name_input if thread_name_input else f"Thread_{new_thread_id}"
                #conversation_data = [{'role': 'system', 'content': mprompts["sys_ragbot"]}]
                conversation_data = [{'role': 'system', 'content': "Ti si expert za pisanje tekstova na srpskom jeziku, narocito u oblasti IT services i digitalne transfomacije. Kreiraj opsirne tekstove iz zadate teme, uvek na srpskom jeziku"}]
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
        ### ovde ide multifile uploader
        image_ai, vrsta = read_txts()
    # Display the existing messages in the current thread
    #  Ensure `messages` and `thread_id` are initialized in session state
    if "messages" not in st.session_state:
        st.session_state.messages = {}
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None
        
    # avatari
    avatar_ai="bot.png" 
    avatar_user = "user.webp"
    avatar_sys= "bot.png"
    ### Main application logic ovde ide iz klotbotnovi
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
                st.session_state.filtered_messages = ""
                filtered_data = [entry for entry in st.session_state.messages[current_thread_id] if entry['role'] in ["user", 'assistant']]
                for item in filtered_data:  # lista za download conversation
                    st.session_state.filtered_messages += (f"{item['role']}: {item['content']}\n")  

                for message in st.session_state.messages[current_thread_id]:
                    if message["role"] == "assistant": 
                        with st.chat_message(message["role"], avatar=avatar_ai):
                             st.markdown(message["content"])
                             copy_to_clipboard(message["content"])
                    elif message["role"] == "user":         
                        with st.chat_message(message["role"], avatar=avatar_user):
                             st.markdown(message["content"])
                    elif message["role"] == "system":
                            pass
                    else:        
                        with st.chat_message(message["role"], avatar=avatar_sys):
                             st.markdown(message["content"])
               
        # Handle new user input
        if prompt := st.chat_input("Kako vam mogu pomoci?"):
            context = image_ai  # ovde ce ici iz uploaded files iz kotbotnovi            
            complete_prompt = prompt + " " + context
        
            # Append user prompt to the conversation
            st.session_state.messages[current_thread_id].append({"role": "user", "content": complete_prompt})
        
            # Display user prompt in the chat
            with st.chat_message("user", avatar=avatar_user):
                st.markdown(prompt)
               
            # Generate and display the assistant's response
            with st.chat_message("assistant", avatar=avatar_ai):
                message_placeholder = st.empty()
                full_response = ""
                for response in client.chat.completions.create(
                model=st.session_state["openai_model"],
                temperature=st.session_state.temp,
                messages=st.session_state.messages[current_thread_id],
                stream=True,
            ):
                    full_response += (response.choices[0].delta.content or "")
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            copy_to_clipboard(full_response)
            st.session_state.messages[current_thread_id].append({"role": "assistant", "content": full_response})
            st.session_state.filtered_messages = ""
            filtered_data = [entry for entry in st.session_state.messages[current_thread_id] if entry['role'] in ["user", 'assistant']]
            for item in filtered_data:  # lista za download conversation
                st.session_state.filtered_messages += (f"{item['role']}: {item['content']}\n")  
            # Append assistant's response to the conversation
          
         
            # Update the conversation in the database
            with ConversationDatabase() as db:
                db.update_sql_record(app_name, st.session_state.username, current_thread_id, st.session_state.messages[current_thread_id])
        with st.sidebar:
            st.download_button(
                        "⤓ Preuzmi", 
                        st.session_state.filtered_messages, 
                        file_name="istorija.txt", 
                        help = "Čuvanje zadatog prompta"
                        )
       
deployment_environment = os.environ.get("DEPLOYMENT_ENVIRONMENT")
 ### ovde ide iz kontrole tokena
if deployment_environment == "Streamlit":
    name, authentication_status, username = positive_login(main, " ")
else:
    ### ovde ide iz kontrole tokena
    if __name__ == "__main__":
        main()