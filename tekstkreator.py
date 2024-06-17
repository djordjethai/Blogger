
import os, uuid
import streamlit as st
from openai import OpenAI
from myfunc.asistenti import read_aad_username
from myfunc.mojafunkcija import (
    positive_login,
    show_logo, initialize_session_state, 
    check_openai_errors, 
    read_txts, 
    copy_to_clipboard)
from myfunc.pyui_javascript import chat_placeholder_color
from myfunc.prompts import ConversationDatabase, PromptDatabase
from myfunc.varvars_dicts import work_vars, work_prompts

client = OpenAI()
mprompts = work_prompts()

def sidebar_width(width):
    st.markdown(
        f"""
        <style>
            section[data-testid="stSidebar"] {{
                width: {width} !important; # Set the width to your desired value
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    
sidebar_width(550)

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
    "app_name": "TekstKreator",
    "upload_key": 0,
}

initialize_session_state(default_values)
chat_placeholder_color("#f1f1f1")

if st.session_state.thread_id not in st.session_state.messages:
    st.session_state.messages[st.session_state.thread_id] = [{'role': 'system', 'content': mprompts["sys_blogger"]}]

if "temp" not in st.session_state:
    st.session_state.temp=0.0


def add_template(kw):
    # Condition filter for prompts
    with PromptDatabase() as db:
        prompt_details = db.get_prompts_contain_in_name(kw)
        template_list = [detail["PromptString"] for detail in prompt_details]    
    with st.sidebar:
        with st.popover("Izaberite Prompt Template"):
            return st.radio("Izaberite tekst", template_list, index=None) 
            
    
def allow_template_value(default_chat_input_value):    
    js = f"""
        <script>
            function insertText(dummy_var_to_force_repeat_execution) {{
                var chatInput = parent.document.querySelector('textarea[data-testid="stChatInputTextArea"]');
                var nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
                nativeInputValueSetter.call(chatInput, "{default_chat_input_value}");
                var event = new Event('input', {{ bubbles: true}});
                chatInput.dispatchEvent(event);
            }}
            insertText({len(st.session_state.messages)});
        </script>
        """
    st.components.v1.html(js)
       
st.subheader("Blogger - kreira tekstove")
st.caption("17.06.24")
with st.expander("Uputstvo"):
    st.caption("### Upotreba Kreatora Tekstova \n" 
               " Bloggera mogu da koriste samo logovani korsinici sa pravom pristupa.")
    st.caption(" Na osnovu uploadovanih dokumenata zadajte instrukcije za pisanje teksta." 
               " Sa strane mozete podesiti temperaturu u zavisnosti od toga koliko kreativan odgovor zelite. \n\n"
               " Nakon prve verzije mozete traziti ispravke. Uvek mozete nastaviti istu konverzaciju kasnije." 
               " Mozete sacuvati konverzaciju ili konkretan odgovor."
               " U ovoj verziji Asistent nema pristup ni internetu ni internim podacima kompanije!")
    st.caption(" Mozete koristiti biblioteku instrukcija.")

def main():
    if "username" not in st.session_state:
        st.session_state.username = "positive"
    if deployment_environment == "Azure":    
        st.session_state.username = read_aad_username()
    elif deployment_environment == "Windows":
        st.session_state.username = "lokal"
    elif deployment_environment == "Streamlit":
        st.session_state.username = username
    conversation_data = [{'role': 'system', 'content': mprompts["sys_blogger"]}]
    
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
       
        st.session_state.temp=st.slider("Temperatura", 0.0, 1.0, 0.0, 0.1)
        with st.popover("Kreirajte novi ili učitajte postojeći razgovor"):
            operation = st.selectbox("Izaberite", ["Novi razgovor", "Učitajte postojeći", "Obrišite"])

            if operation == "Novi razgovor":
                st.caption("Kreirajte novi razgovor")
                thread_name_input = st.text_input("Unesite ime razgovora (opciono) i POTVRDITE SA ENTER!")

                if st.button("Kreiraj"):
                    new_thread_id = str(uuid.uuid4())
                    thread_name = thread_name_input if thread_name_input else f"Thread_{new_thread_id}"
                    conversation_data = [{'role': 'system', 'content': mprompts["sys_blogger"]}]
                    
                    # Check if the thread ID already exists
                    if thread_name not in get_thread_ids():
                        with ConversationDatabase() as db:
                            db.add_sql_record(app_name, st.session_state.username, thread_name, conversation_data)
                        st.success(f"Razgovor {thread_name} je uspešno kreiran.")
                        st.session_state.thread_id = thread_name
                    else:
                        st.error("Razgovor već postoji.")
                            
            elif operation == "Učitajte postojeći":
                    st.caption("Učitajte postojeći razgovor")
                    thread_ids = get_thread_ids()
                    selected_thread_id = st.selectbox("Odaberite razgovor", thread_ids)

                    if st.button("Odaberi"):
                        with ConversationDatabase() as db:
                            db.query_sql_record(app_name, st.session_state.username, selected_thread_id)
                        st.success(f"Razgovor uspešno učitan {selected_thread_id}.")
                        st.session_state.thread_id =  selected_thread_id

            elif operation == "Obrišite":
                st.caption("Obrišite postojeći razgovor")
                thread_ids = get_thread_ids()
                selected_thread_id = st.selectbox("Odaberite razgovor", thread_ids)

                if st.button("Obriši"):
                    try:
                        with ConversationDatabase() as db:
                            db.delete_sql_record(app_name, st.session_state.username, selected_thread_id)
                        st.success(f"Razgovor {selected_thread_id} je uspešno obrisan.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Nisam uspeo da obrišem razgovor: {e}")

    with st.sidebar:
        ### ovde ide multifile uploader
        with st.popover("Učitajte dokumente za kreiranje teksta"):
            image_ai, _ = read_txts()
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
        st.info("Započnite razgovor učitavanjem postojećeg ili kreiranjem novog.")
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
                             copy_to_clipboard(message["content"])
                    elif message["role"] == "system":
                            pass
                    else:        
                        with st.chat_message(message["role"], avatar=avatar_sys):
                             st.markdown(message["content"])
               
        # Handle new user input
        full_template = add_template("blog_template") 
        if full_template:
            allow_template_value(full_template)                        
        if prompt := st.chat_input("Opišite dokument koji želite da kreiram"):
            if image_ai:
                context = image_ai
            else:
                context = ""
            complete_prompt = prompt + " " + context
            
            # Append user prompt to the conversation
            st.session_state.messages[current_thread_id].append({"role": "user", "content": complete_prompt})
            # Display user prompt in the chat
            with st.chat_message("user", avatar=avatar_user):
                st.markdown(prompt)
                copy_to_clipboard(prompt)
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
            # Update the conversation in the database
            with ConversationDatabase() as db:
                db.update_sql_record(app_name, st.session_state.username, current_thread_id, st.session_state.messages[current_thread_id])
        with st.sidebar:
            st.download_button(
                        "⤓ Preuzmi", 
                        st.session_state.filtered_messages, 
                        file_name="istorija.txt", 
                        help = "Čuvanje istorije ovog razgovora"
                        )
       
deployment_environment = os.environ.get("DEPLOYMENT_ENVIRONMENT")
 ### ovde ide iz kontrole tokena
if deployment_environment == "Streamlit":
    name, authentication_status, username = positive_login(main, " ")
else:
    ### ovde ide iz kontrole tokena
    if __name__ == "__main__":
         check_openai_errors(main)