
import ai21
import json as json
import os
import streamlit as st 

USER_ICON = "images/user-icon.png"
AI_ICON = "images/ai-icon.png"
SUMMARIZER_ENDPOINT_NAME = "ai21-summarizer-endpoint"

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
                    font-size: 24px;
                }
        </style>
        """, unsafe_allow_html=True)

def write_logo():
    col1, col2, col3 = st.columns([1, 1, 5])
    with col1:
        st.image(AI_ICON, use_column_width='always')
    with col3:
        header = f"Generative AI Powered Business Document Summarizer!"
        st.write(f"<h3 class='main-header'>{header}</h3>", 
                 unsafe_allow_html=True)

def write_top_bar():
    col1, col2, col3 = st.columns([12,1,4])
    with col1:
        selected_doc = st.selectbox( 
        'Please choose a Document',
         ('Amazon Shareholder Letter 2022', 
          'Amazon Shareholder Letter 2021'))
    with col2:
        pass
    with col3:
        selected_page = st.selectbox( 
            'Page Number',
            ('1', 
            '2',
            '3',
            '4',
            '5',
            '6',
            '7',
            '8'))
    return selected_doc, selected_page

def get_text_source_file(selected_doc, selected_page):
    if selected_doc == 'Amazon Shareholder Letter 2022':
        filepath = f'docs/2022-Shareholder-Letter'
    elif selected_doc == 'Amazon Shareholder Letter 2021':
        filepath = f'docs/2021-Shareholder-Letter'
    filepath = f'{filepath}_{selected_page}.txt'
    return filepath

def generate_summary(selected_doc, selected_page):
    source_file = get_text_source_file(selected_doc, selected_page)
    print(source_file)
    with open(source_file, 'r') as f:
        source_text = f.read()
    response = ai21.Summarize.execute(
                          source=source_text,
                          sourceType="TEXT",
                          destination=ai21.SageMakerDestination(SUMMARIZER_ENDPOINT_NAME)
    )
    summary_results = st.text_area(label="summary",
        value=f"{response.summary}",
        key="summary_results",
        label_visibility="hidden",
        height=640)
    
if __name__ == "__main__":
    write_logo()
    selected_doc, selected_page  = write_top_bar()
    st.markdown('---')
    header=f"Summary of page {selected_page} of {selected_doc}"
    col1, col2, col3 = st.columns([2,12,1])
    with col2:
        prompt = st.button(f"Click here to generate {header}",
                type="primary")
    if not prompt:
        st.text_area(label="summary",
                    value="Summary will be shown here",
                    label_visibility="hidden",
                    height=240)
    else:
        summary = generate_summary(selected_doc, selected_page)
    
