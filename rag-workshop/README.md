# Prequisites

- A quota of 1+ for following instance types
-- ml.g5.12xlarge (to deploy encoder LLM model)


# Step 1 : Deploy OpenSearch Cluster
## Insturctions to create Opensearch clutser using Cloudformation scripts 
### Steps 
    - Goto Cloudformation service on AWS console
    - Click on create stack
    - Choose "Specify template" and Upload create-opensearch.yaml file 
    - provide a stack name such as opensearch-embedding-stack
    - provide OpenSearchIndexName sch as pdfdoc_embeddings_vectors
    - provide password 
    - leave OpenSearchUsername unchanged
    - leave options unchanged and click next
    - Choose submit  
At this point, the cloudformation script will start the deployment of OpenSearch Cluster. If there are any errors, the entire deployment process will be rolledback automatically. The process will take 10-15 mins.

# Step 2 : Setup Kendra
    1. Create Kendra Index
    - Upload PDF files to your S3 bucket 
    - Goto Amazon Kendra Console and select 'create index'
    - Specify following values
        'Index name' : shareholder-letters-index
        'IAM Role': Select 'Create a new role  (recommended)'
    - click Next 
    - Leave default options and click Next
    - Select Developer Edit and click 'Create'
    Note: It will take 15-30 minutes for Kendra to create Index 
    
    2. Use S3 connector to add 
    - Select the Index you just created 
    - Click 'Add Data Source'
    - Select 'Add connector' for 'Amazon S3 Connector' on the next screen
    - Give data source a name 'shareholder-letters-pdf'
    - leave other options unchange and click 'Next'
    - Select 'create a new role' for IAM and give it appropriate name 
    - Leave 'No VPC' in the Configure VPC section
    - Click Next
    - Select the bucket containing PDFs for the data bucket location under the 'Sync Scope' 
    - Choose 'Run on demand' for Frequency
    - leave other options as-is
    - Click Next
    - Click Next on the 'Set Mappings optional' screen
    - Click 'Add Source' (final screen)
    - Once data source has been added, click 'sync now' option on the data source screen
    Note: It will take few minutes for Kendra to crawl and index the documents 
    

# Step3 : : Deploy Encoder and Caual LLM Model 
#### While OpenSearch clutser is being created, 
- Run the 'deploy_llms.ipynb' notebook


## Before continuing, ensure that 
    - LLM Models are deployed
    - Kendra is done syncing the data source
    - OpenSearch index has been created successfully

# Step 4: Validate Kendra indexer is working properly 
#### Run 'kendra-based-qna' notebook


# Step 5: Implement FAISS based local indexer
#### Run 'FAISS-based-qna' notebook


# Step 6: Use OpenSearch as vectordb for RAG
#### Run 'Opensearch-based-qna' notebook

# Step 7: Run Streamlit based UI for QnA bot 
    1. Open System terminal through launcher
    2. Ensure current working directory is rag-workshop
    3. Run `pip install streamlit "ai21[AWS]" langchain faiss-cpu  opensearch-py`
    3. Run `sh setup.sh`
    4. Run `sh run.sh` 
        Follow the URL for the streamlit app 


