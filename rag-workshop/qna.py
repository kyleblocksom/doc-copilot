import streamlit as st 
import sys
import os
import uuid
import ai21 

import sagemaker
import boto3
from langchain.retrievers import AmazonKendraRetriever
from langchain.vectorstores import FAISS
from load_embeddings import create_sagemaker_embeddings_from_js_model, encoder_name, encoder_endpoint_name
from langchain.vectorstores import OpenSearchVectorSearch
from opensearch import get_stack_details, get_credentials, opensearch_index_name
import json

USER_ICON = "images/user-icon.png"
AI_ICON = "images/ai-icon.png"
MAX_HISTORY_LENGTH = 5

## initialize credentials and boto3 client
boto3_session = boto3.session.Session()
aws_region = boto3_session.region_name

PROVIDER_MAP = {
    'Amazon Kendra': 'Kendra',
    'FAISS Indexer' : 'FAISS',
    'Amazon OpenSearch' : 'OpenSearch'
}


# initialize kendra retriever 
kendra_index_id = '1a1d7832-1658-4cbf-9369-57bacc68f503' ## change it to match your index id 
os.environ['KENDRA_INDEX_ID'] = kendra_index_id
kendra = boto3.client("kendra", aws_region)
# Kendra retriever
retriever = AmazonKendraRetriever(
    index_id=kendra_index_id,
    region_name=aws_region,
    top_k=5
)

# initialize FIASS indexer 
embeddings = create_sagemaker_embeddings_from_js_model(encoder_endpoint_name, aws_region)
docsearch = FAISS.load_local('shareholders_letter.faiss', embeddings)

# initialize opensearch indexer
results = get_stack_details(aws_region=aws_region)
print(results)
creds = get_credentials(results['opensearch_secretid'], aws_region)
http_auth = (creds['username'], creds['password'])
opensearch = OpenSearchVectorSearch(index_name=opensearch_index_name,
                                   embedding_function=embeddings,
                                   opensearch_url=results['opensearch_domain_endpoint'],
                                   http_auth=http_auth)


#### 

# Check if the user ID is already stored in the session state
if 'user_id' in st.session_state:
    user_id = st.session_state['user_id']
# If the user ID is not yet stored in the session state, generate a random UUID
else:
    user_id = str(uuid.uuid4())
    st.session_state['user_id'] = user_id

# Initialize chat history 
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Initialize chats array to hold 5 previous interactions
if "chats" not in st.session_state:
    st.session_state.chats = [
        {
            'id': 0,
            'question': '',
            'answer': ''
        }
    ]

# initialize questions array 
if "questions" not in st.session_state:
    st.session_state.questions = []

# Initialize answers array 
if "answers" not in st.session_state:
    st.session_state.answers = []

# Initialize inputs array 
if "input" not in st.session_state:
    st.session_state.input = ""

st.markdown("""
        <style>
               .block-container {
                    padding-top: 32px;
                    padding-bottom: 32px;
                    padding-left: 0;
                    padding-right: 0;
                }
                .element-container img {
                    background-color: #000000;
                }

                .main-header {
                    font-size: 32px;
                }
                .main-subheader {
                    font-size: 16px;
                }
        </style>
        """, unsafe_allow_html=True)

def write_logo():
    col1, col2, col3 = st.columns([5, 1, 5])
    with col2:
        st.image(AI_ICON, use_column_width='always') 

def clear_chat():
    st.session_state.questions = []
    st.session_state.answers = []
    st.session_state.input = ""
    st.session_state["chat_history"] = []

def write_top_bar():
    col1, col2, col3 = st.columns([1,10,2])
    with col1:
        st.image(AI_ICON, use_column_width='always')
    with col2:
        header = f"QnA App!"
        st.write(f"<h3 class='main-header'>{header}</h3>", 
                 unsafe_allow_html=True)
        selected_provider = st.selectbox( 
        'Please choose a Search engine',
         ('Amazon Kendra',
          'FAISS Indexer',
          'Amazon OpenSearch'
          ),
          on_change=clear_chat,
          )
        if selected_provider in PROVIDER_MAP:
            provider = PROVIDER_MAP[selected_provider]
        else:
            provider = selected_provider.capitalize()
        header = f"Powered by {selected_provider}!"
        st.write(f"<h4 class='main-subheader'>{header}</h4>", 
                 unsafe_allow_html=True)
    with col3:
        clear = st.button("Clear Chat", on_click=clear_chat)
    return clear, provider


def opensearch_handle_input():
    model_name = "contextual-answers"
    endpoint_name = f'{model_name}-endpoint'
    input = st.session_state.input
    question_with_id = {
        'question': input,
        'id': len(st.session_state.questions)
    }
    st.session_state.questions.append(question_with_id)

    chat_history = st.session_state["chat_history"]
    if len(chat_history) == MAX_HISTORY_LENGTH:
        chat_history = chat_history[:-1]
    docs = opensearch.similarity_search(input, k=3, include_metadata=False)
    context = docs[0].page_content + docs[1].page_content + docs[2].page_content
    print('OpenSearch context : ', context)
    response = ai21.Answer.execute(
        context=context,
        question=input,
        destination=ai21.SageMakerDestination(endpoint_name)
    )
    st.session_state.answers.append({
        'answer': response.answer,
        'id': len(st.session_state.questions)
    })
    st.session_state.input = ""


def faiss_handle_input():
    model_name = "contextual-answers"
    endpoint_name = f'{model_name}-endpoint'
    input = st.session_state.input
    question_with_id = {
        'question': input,
        'id': len(st.session_state.questions)
    }
    st.session_state.questions.append(question_with_id)

    chat_history = st.session_state["chat_history"]
    if len(chat_history) == MAX_HISTORY_LENGTH:
        chat_history = chat_history[:-1]
    docs = docsearch.similarity_search(input, k=3)
    context = docs[0].page_content + docs[1].page_content + docs[2].page_content
    print('FIASS context : ', context)
    response = ai21.Answer.execute(
        context=context,
        question=input,
        destination=ai21.SageMakerDestination(endpoint_name)
    )
    st.session_state.answers.append({
        'answer': response.answer,
        'id': len(st.session_state.questions)
    })
    st.session_state.input = ""

def handle_input():
    model_name = "contextual-answers"
    endpoint_name = f'{model_name}-endpoint'
    input = st.session_state.input
    question_with_id = {
        'question': input,
        'id': len(st.session_state.questions)
    }
    st.session_state.questions.append(question_with_id)

    chat_history = st.session_state["chat_history"]
    if len(chat_history) == MAX_HISTORY_LENGTH:
        chat_history = chat_history[:-1]


    docs = retriever.get_relevant_documents(input)
    context = docs[0].page_content
    print('Kendra context : ', context)

    response = ai21.Answer.execute(
        context=context,
        question=input,
        destination=ai21.SageMakerDestination(endpoint_name)
    )
    st.session_state.answers.append({
        'answer': response.answer,
        'id': len(st.session_state.questions)
    })
    st.session_state.input = ""

def write_user_message(md):
    col1, col2 = st.columns([1,12])
    
    with col1:
        st.image(USER_ICON, use_column_width='always')
    with col2:
        st.warning(md['question'])


def render_result(result):
    answer, sources = st.tabs(['Answer', 'Sources'])
    with answer:
        render_answer(result['answer'])
    with sources:
        if 'source_documents' in result:
            render_sources(result['source_documents'])
        else:
            render_sources([])

def render_answer(answer):
    col1, col2 = st.columns([1,12])
    with col1:
        st.image(AI_ICON, use_column_width='always')
    with col2:
        st.info(answer)

def render_sources(sources):
    col1, col2 = st.columns([1,12])
    with col2:
        with st.expander("Sources"):
            for s in sources:
                st.write(s)

    
#Each answer will have context of the question asked in order to associate the provided feedback with the respective question
def write_chat_message(md, q):
    chat = st.container()
    with chat:
        render_answer(md['answer'])
        #render_sources(md['sources'])
    
if __name__ == "__main__":
    clear, provider = write_top_bar()

    if clear:
        st.session_state.questions = []
        st.session_state.answers = []
        st.session_state.input = ""
        st.session_state["chat_history"] = []
    with st.container():
        for (q, a) in zip(st.session_state.questions, st.session_state.answers):
            write_user_message(q)
            write_chat_message(a, q)

    st.markdown('---')
    if provider == 'Kendra':
        input = st.text_input("You are talking to Kendra powered AI, Please ask a question in context of document(s).", 
                            key="input", 
                            on_change=handle_input)
    elif provider == 'FAISS':
            input = st.text_input("You are talking to FAISS powered AI, Please ask a question in context of document(s).", 
                        key="input", 
                        on_change=faiss_handle_input)
    elif provider == 'OpenSearch':
            input = st.text_input("You are talking to OpenSearch powered AI, Please ask a question in context of document(s).", 
                        key="input", 
                        on_change=opensearch_handle_input)
