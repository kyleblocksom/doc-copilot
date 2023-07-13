import os
import json
import time
import base64
import pandas as pd
import streamlit as st
from PIL import Image
from trp import Document
import trp.trp2 as t2

import boto3
import botocore
from textractcaller import QueriesConfig, Query
from textractcaller.t_call import call_textract, Textract_Features, call_textract_expense
from textractprettyprinter.t_pretty_print import convert_table_to_list, Pretty_Print_Table_Format, Textract_Pretty_Print, get_string, convert_queries_to_list_trp2

from langchain.agents.tools import Tool
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage
)

from langchain.document_loaders import S3DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import NLTKTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.schema import Document
#import sagemaker

from langchain import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import Dict


# variables
data_bucket = "YOUR-BUCKET-NAME" # st.secrets["data_bucket"]
access_key_id = "YOUR-ACCESS-KEY-ID" # st.secrets["access_key_id"]
secret_access_key = "YOUR-ACCESS-KEY" # st.secrets["secret_access_key"]
region = "us-east-1"

# files
filename = "amazon-sec-demo.pdf" 
file = "../doc_sample/amazon-sec-demo.pdf" # "../doc_sample/grant-deed.pdf"
file2 = "s3://genai-bucket/amazon-sec-demo.pdf" # "../doc_sample/genai-demo-doc.pdf" #st.secrets["file"]
idp_logo = "idp-logo.png"

# boto3 clients
s3=boto3.client('s3')
textract = boto3.client('textract', region_name=region)

newline, bold, unbold = '\n', '\033[1m', '\033[0m'
summarization_endpoint_name = 'jumpstart-dft-hf-summarization-distilbart-xsum-1-1'

# Custom SageMaker Endpoint LangChain LLM class
class QAContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"text_inputs": prompt, **model_kwargs})
        return input_str.encode('utf-8')
    
    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["generated_texts"][0]

#helper functions
def show_pdf(file_path):
    with open(file_path,"rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="500" height="500" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


def generate_chunks(inp_str):
    max_chunk = 500
    inp_str = inp_str.replace('.', '.<eos>')
    inp_str = inp_str.replace('?', '?<eos>')
    inp_str = inp_str.replace('!', '!<eos>')
    
    sentences = inp_str.split('<eos>')
    current_chunk = 0 
    chunks = []
    for sentence in sentences:
        if len(chunks) == current_chunk + 1: 
            if len(chunks[current_chunk]) + len(sentence.split(' ')) <= max_chunk:
                chunks[current_chunk].extend(sentence.split(' '))
            else:
                current_chunk += 1
                chunks.append(sentence.split(' '))
        else:
            chunks.append(sentence.split(' '))

    for chunk_id in range(len(chunks)):
        chunks[chunk_id] = ' '.join(chunks[chunk_id])
    return chunks


def startJob(s3BucketName, objectName):
    response = None
    response = textract.start_document_text_detection(
    DocumentLocation={
        'S3Object': {
            'Bucket': s3BucketName,
            'Name': objectName
        }
    })

    return response["JobId"]


def isJobComplete(jobId):
    response = textract.get_document_text_detection(JobId=jobId)
    status = response["JobStatus"]
    print("Job status: {}".format(status))

    while(status == "IN_PROGRESS"):
        time.sleep(3)
        response = textract.get_document_text_detection(JobId=jobId)
        status = response["JobStatus"]
        print("Job status: {}".format(status))

    return status


def getJobResults(jobId):

    pages = []
    response = textract.get_document_text_detection(JobId=jobId)
    
    pages.append(response)
    print("Resultset page recieved: {}".format(len(pages)))
    nextToken = None
    if('NextToken' in response):
        nextToken = response['NextToken']

    while(nextToken):
        response = textract.get_document_text_detection(JobId=jobId, NextToken=nextToken)

        pages.append(response)
        print("Resultset page recieved: {}".format(len(pages)))
        nextToken = None
        if('NextToken' in response):
            nextToken = response['NextToken']

    return pages


def query_endpoint(endpoint_name, encoded_text):
    client = session.client('runtime.sagemaker')
    response = client.invoke_endpoint(EndpointName=endpoint_name, ContentType='application/x-text', Body=encoded_text)
    return response


def query_endpoint_with_json_payload(endpoint_name, encoded_json):
    client = session.client("runtime.sagemaker")
    response = client.invoke_endpoint(
        EndpointName=endpoint_name, ContentType="application/json", Body=encoded_json
    )
    return response


def parse_response(query_response):
    model_predictions = json.loads(query_response['Body'].read())
    generated_text = model_predictions['generated_text']
    return generated_text


def parse_response_multiple_texts(query_response):
    model_predictions = json.loads(query_response["Body"].read())
    generated_text = model_predictions["generated_texts"]
    return generated_text


# --- Main ---


session = boto3.session.Session(aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key, region_name=region)
textract = session.client('textract', region_name=region)
comprehend = session.client('comprehend', region_name=region)
prefix = 'idp/genai/'

st.title('GenAI Document Co-Pilot')
st.subheader('Powered by coffee and AWS AI/ML Services')


st.info("**DISCLAIMER:** This demo uses a Bedrock foundational model and not intended to collect any personally identifiable information (PII) from users. Please do not provide any PII when interacting with this demo. The content generated by this demo is for informational purposes only.")
           
st.sidebar.image(idp_logo, width=300, output_format='PNG')
st.sidebar.subheader('**About this Demo**')

st.sidebar.success("**GenAI Document Co-Pilots** is a Generative AI and AWS IDP-based document assistant that can quickly extract data, categorize documents, extract insights, summarize, and have conversations with any types of documents.") 


with st.expander("Sample PDF 📁"):
    show_pdf(file)

st.subheader('Document Classification')

if st.button('Classify the Sample'):
    text = []

    jobId = startJob(data_bucket, filename)
    if(isJobComplete(jobId)):
        response = getJobResults(jobId)

    # Append detected text
    for resultPage in response:
        for item in resultPage["Blocks"]:
            if item["BlockType"] == "LINE":
                text.append(item["Text"])

    textract_text = "\n".join(text)

    # Detect PII   
    response = comprehend.detect_pii_entities(
        Text= textract_text,
        LanguageCode='en'
    )

    # Document Label
    prompt_text = "Given the following text, what is the document type for this text? %s"%(textract_text)
    comprehend_txt = textract_text
    for text in [prompt_text]:
        query_response = query_endpoint(summarization_endpoint_name, json.dumps(text).encode('utf-8'))
        generated_text = parse_response(query_response)
        
        st.write("✅ **Document Label:**",generated_text)

    for entity in reversed(response['Entities']):
        comprehend_txt  = textract_text[:entity['BeginOffset']] + entity['Type'] + comprehend_txt[entity['EndOffset']:]

    st.subheader("🔐**Document De-identification**")
    st.text_area("Scroll down:",comprehend_txt)

    parameters = {
        "max_length": 100,
        "top_k": 50,
        "top_p": 0.95,
        "do_sample": True,
    }

    summ_text = "Given the following text, summarize the document? %s"%(textract_text)

# Table extraction with Textract 
st.subheader('Structured and Semi-structured data extraction')
if st.button('Extract tables and forms'):   
    resp = call_textract(input_document=file2, features=[Textract_Features.TABLES, Textract_Features.FORMS], boto3_textract_client=textract)
    tdoc = Document(resp)
    dfs = list()

    kvlist = get_string(textract_json=resp,
            table_format=Pretty_Print_Table_Format.fancy_grid,
            output_type=[Textract_Pretty_Print.FORMS])

    st.text(kvlist)

    for page in tdoc.pages:
        for table in page.tables:
            tab_list = convert_table_to_list(trp_table=table)
            dfs.append(pd.DataFrame(tab_list))

    for df in dfs:
        header_row = df.iloc[0]
        df2 = pd.DataFrame(df.values[1:], columns=header_row)  
        st.text("Tables:") 
        st.dataframe(df2)

# Document Summarization      
st.subheader('Document Summarization')

if st.button('Summarize'):
    text = []

    jobId = startJob(data_bucket, filename)
    print("Started job with id: {}".format(jobId))
    if(isJobComplete(jobId)):
        response = getJobResults(jobId)

    # Append detected text
    for resultPage in response:
        for item in resultPage["Blocks"]:
            if item["BlockType"] == "LINE":
                text.append(item["Text"])

    textract_text = "\n ".join(text)
  
    txt = st.text_area('Text to analyze', textract_text)
    print('Text to analyze = ' + textract_text)
    query_response = query_endpoint(summarization_endpoint_name, json.dumps(textract_text).encode('utf-8'))
    summary_text = parse_response(query_response)
    print('Summary text = ' + summary_text)

    #time.sleep(5)
    st.write("✅ **Summary:**",summary_text)

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False
    
# Document Q & A 
st.subheader('Document Question & Answering')

question = st.text_input('Enter your question here')

if len(question) > 0:
    embeddings = HuggingFaceEmbeddings()
    loader = S3DirectoryLoader(data_bucket, prefix='grant-deed.pdf')

    # Permissions issue
    docs = loader.load() 
    text_splitter = NLTKTextSplitter(chunk_size=550)
    texts = text_splitter.split_documents(docs)
    vectordb = Chroma.from_documents(texts, embeddings)

    FLAN_T5_PARAMETERS = {
        "temperature": 0.97,           # the value used to modulate the next token probabilities.
        "max_length": 100,             # restrict the length of the generated text.
        "num_return_sequences": 3,     # number of output sequences returned.
        "top_k": 50,                   # in each step of text generation, sample from only the top_k most likely words.
        "top_p": 0.95,                 # in each step of text generation, sample from the smallest possible set of words with cumulative probability top_p.
        "do_sample": True              # whether or not to use sampling; use greedy decoding otherwise.
    }

    qa_content_handler = QAContentHandler()
    prompt_template="""Given the following text from a document, answer the question to the best of your abilities. Answer only from the provided document,, if you do not know the answer 
    just say you don't know. DO NOT make up an answer.

    Document: {document}
    Question: {question}
    Answer:
    """

    prompt=PromptTemplate(input_variables=["document", "question"], 
                                                   template=prompt_template)
    qa_endpoint_name = 'jumpstart-dft-hf-text2text-flan-t5-xl'

    qa_chain = LLMChain(
        llm=SagemakerEndpoint(
            endpoint_name=qa_endpoint_name, # replace with your endpoint name if needed
            region_name=region,
            model_kwargs=FLAN_T5_PARAMETERS,
            content_handler=qa_content_handler
        ),
        prompt=prompt
    )

    similar_docs = vectordb.similarity_search(question, k=3) #see also : max_marginal_relevance_search_by_vector(query, k=3)
    context_list = [a.page_content for a in similar_docs]
    metadata_list = [a.metadata.get('source') for a in similar_docs]
    context = "\n\n".join(context_list)

    answer = qa_chain.run({
        'document': context,
        'question': question
    })

    #time.sleep(5)
    st.write("✅ **Answer:**",answer)
