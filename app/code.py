
from flask import Blueprint, render_template, request
import openai
import requests
import json
import pandas as pd
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import tensorflow_hub as hub
import faiss
import torch.nn as nn
import re
import requests

code = Blueprint('code', __name__)

@code.route('/', methods=['GET', 'POST'])
def index():
    #Write Code Here
    #testing
    result = ""

    if request.method == 'POST':
        #Get values from form
        Medication = request.form['Medication']
        Disease = request.form['Disease']
        Age = request.form['Age']
        Sex = request.form['Sex']
        Ethnicity = request.form['Ethnicity']

        # Initialize the results dictionary
        results_dict = {
            "Clinical_trial_data": "",
            "Relevant_ADR_reports": "",
            "ADR_statistics": ""
        }

        #EXTRACT THE CLINICAL TRIALS FOR A SPECIFIC MEDICATION AND DISEASE
        # Define the payload for the request
        payload = {
            "query.cond": Disease,
            "query.term": Medication,
            "filter.overallStatus": 'COMPLETED',
            "query.intr": Medication,
        }

        # Define the URL and make the request
        url = "https://clinicaltrials.gov/api/v2/studies"
        response = requests.get(url, params=payload)

        # Define the DotDict class
        class DotDict(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__
            __delattr__ = dict.__delitem__

            def __init__(self, dct):
                for key, value in dct.items():
                    if hasattr(value, 'keys'):
                        value = DotDict(value)
                    self[key] = value

        # Initialize an empty dictionary to store the results
        results_dict = {}
        trial_ids = []

        # Check if the request was successful
        if response.status_code == 200:
            try:
                data = response.json()
                studies = data.get('studies', [])

                # Collect clinical trial data as a string
                clinical_trial_data = []

                for idx, study_data in enumerate(studies):
                    study = DotDict(study_data)
                    try:
                        # Get the study number (nctId)
                        nct_id = study.protocolSection.identificationModule.nctId

                        # Check if the required nested attributes are present
                        if (
                            hasattr(study, 'resultsSection') and
                            hasattr(study.resultsSection, 'baselineCharacteristicsModule') and
                            hasattr(study.resultsSection.baselineCharacteristicsModule, 'measures')
                        ):
                            # Add the study number and clinical trial data
                            clinical_trial_data.append(f"Study {nct_id}:\n")
                            clinical_trial_data.append(f"{study.protocolSection.eligibilityModule}\n")
                            clinical_trial_data.append(f"{study.resultsSection.baselineCharacteristicsModule.measures}\n\n")
                            trial_ids.append({nct_id})
                    except (KeyError, AttributeError):
                        continue

                # Save clinical trial data as a string in results_dict
                results_dict["Clinical_trial_data"] = ''.join(clinical_trial_data)


            except requests.exceptions.JSONDecodeError:
                print("Failed to decode JSON. Here is the raw response:")
                print(response.text)
        else:
            print(f"Request failed with status code: {response.status_code}")
            print(response.text)

        #GET THE ADR REPORT STATISTICS FOR SPECIFIC MEDICATION 
        #NEED MEDICATION EMBEDDINGS *******
       
        # File path
        path_ADRdata = '/content/drive/My Drive/Programs/BOREALISAI/RAGModel/Data/ADRdata.csv'

        result = "Hello, World!"
        return render_template('home.html', result=result)
    
    return render_template('home.html')

