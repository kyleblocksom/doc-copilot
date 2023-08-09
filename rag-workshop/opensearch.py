import sys
import logging
import boto3
from typing import List
import json

CFN_STACK_NAME = "opensearch-embedding-stack"
opensearch_index_name = f"pdfdoc_embeddings_vectors"

def get_credentials(secret_id: str, region_name: str) -> str:
    
    client = boto3.client('secretsmanager', region_name=region_name)
    response = client.get_secret_value(SecretId=secret_id)
    secrets_value = json.loads(response['SecretString'])    
    return secrets_value


def get_cfn_outputs(stackname: str, aws_region: str) -> List:
    cfn = boto3.client('cloudformation', region_name=aws_region)
    outputs = {}
    for output in cfn.describe_stacks(StackName=stackname)['Stacks'][0]['Outputs']:
        outputs[output['OutputKey']] = output['OutputValue']
    return outputs

def get_cfn_parameters(stackname: str, aws_region: str) -> List:
    cfn = boto3.client('cloudformation', region_name=aws_region)
    params = {}
    for param in cfn.describe_stacks(StackName=stackname)['Stacks'][0]['Parameters']:
        params[param['ParameterKey']] = param['ParameterValue']
    return params

def get_stack_details(aws_region):
    stacks = boto3.client('cloudformation', region_name=aws_region).list_stacks()
    stack_found = CFN_STACK_NAME in [stack['StackName'] for stack in stacks['StackSummaries']]
    print(f'{CFN_STACK_NAME} stack found: {stack_found}')
    if stack_found is True:
        outputs = get_cfn_outputs(CFN_STACK_NAME, aws_region)
        params = get_cfn_parameters(CFN_STACK_NAME, aws_region)
        opensearch_domain_endpoint = f"https://{outputs['OpenSearchDomainEndpoint']}"
        opensearch_domain_name =  outputs['OpenSearchDomainName']
        aws_region = outputs['Region']
        opensearch_secretid = outputs['OpenSearchSecret']
        opensearch_domain_name =  outputs['OpenSourceDomainArn']
        # ARN of the secret is of the following format arn:aws:secretsmanager:region:account_id:secret:my_path/my_secret_name-autoid
        os_creds_secretid_in_secrets_manager = "-".join(outputs['OpenSearchSecret'].split(":")[-1].split('-')[:-1])
        return {
            "opensearch_domain_endpoint" : opensearch_domain_endpoint, 
            'opensearch_domain_name' : opensearch_domain_name, 
            'opensearch_secretid' : opensearch_secretid, 
            'opensearch_domain_name' : opensearch_domain_name, 
            'os_creds_secretid_in_secrets_manager' : os_creds_secretid_in_secrets_manager
        }
