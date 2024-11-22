
from flask import Blueprint, render_template, request
import openai
import requests
import json
import pandas as pd
from requests.adapters import HTTPAdapter, Retry
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

# Set environment variables to resolve OpenMP runtime conflict and disable oneDNN custom operations
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OPENAI_API_KEY"] = "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"

code = Blueprint('code', __name__)

# Function to create a requests session with retries
def requests_retry_session(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

@code.route('/', methods=['GET', 'POST'])
def index():
    #Write Code Here
    result = ""
    trials = ""
    source = ""

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
        path_ADRdata = 'app/ADRdata.csv' #check if this is the correct path

        # Load the CSV file
        df = pd.read_csv(path_ADRdata, delimiter='\t')

        # Define the specific string you're looking for
        specific_string = Medication.upper()

        # Filter rows where the DRUGNAME column contains the specific string
        filtered_df = df[df['DRUGNAME'].str.contains(specific_string, case=False, na=False)]

        # @Vin: Now, replace df with filtered_df because you use `df` variable for stats below
        df = filtered_df

        # ---- GET AGE INFORMATION -----
        # Define age bins and labels for grouping
        age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, float('inf')]
        age_labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100', '100+']

        # Categorize ages into bins
        df['AGE_GROUP'] = pd.cut(df['AGE_Y'], bins=age_bins, labels=age_labels, right=False)

        # Count occurrences for each age group and calculate percentages
        age_group_counts = df['AGE_GROUP'].value_counts().sort_index()
        age_group_percentages = (age_group_counts / age_group_counts.sum()) * 100

        # Select the top 3 age groups by count
        top_3_age_groups = age_group_percentages.nlargest(3).index.astype(str).tolist()
        age_group_sentence = f"The patients with the highest rate of reported drug reaction for this medicine are \"{top_3_age_groups[0]}\" then \"{top_3_age_groups[1]}\" then \"{top_3_age_groups[2]}\"."

        # ----- GET SEX INFORMATION ------
        gender_counts = df['GENDER_ENG'].value_counts()
        gender_percentages = (gender_counts / gender_counts.sum()) * 100

        female_percentage = gender_percentages.get('Female', 0)
        male_percentage = gender_percentages.get('Male', 0)
        gender_sentence = f"Out of all the reports, {female_percentage:.2f}% were women's and {male_percentage:.2f}% were men's."

        # ----- GET SERIOUSNESS INFORMATION -----  POTENTIALLY REMOVE
        seriousness_counts = df['SERIOUSNESS_ENG'].value_counts()
        seriousness_percentages = (seriousness_counts / seriousness_counts.sum()) * 100

        serious_percentage = seriousness_percentages.get('Serious', 0)
        not_serious_percentage = seriousness_percentages.get('Not Serious', 0)
        seriousness_sentence = f"Out of all the reported ADRs, {serious_percentage:.2f}% were serious and {not_serious_percentage:.2f}% were not serious."

        # Combine all sentences into one string and save to ADR_statistics key
        results_dict["ADR_statistics"] = f"{age_group_sentence}\n{gender_sentence}\n{seriousness_sentence}"


        #GET MOST RELEVANT ADR REPORTS
        model_str = "NeuML/pubmedbert-base-embeddings"
        tokenizer = AutoTokenizer.from_pretrained(model_str)
        model = AutoModel.from_pretrained(model_str)

        # ANA - VIN - AIMEE updated november 21st
        def get_sentence_embedding(sentences):
            batch_size = 32
            embeddings = []
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]  # Get a batch of sentences

                encoded_inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
                with torch.no_grad():
                    outputs = model(**encoded_inputs)[0] # get the last hidden state vector
                attention_mask = encoded_inputs['attention_mask'][..., None] # handle padding with attention mask

                sentence_embedding = torch.sum(outputs * attention_mask, dim=1) / torch.clamp(torch.sum(attention_mask, dim=1), min=1e-9)
                norms = torch.clamp(torch.sqrt(torch.sum(sentence_embedding**2, dim=1)), min=1e-9)[..., None]
                sentence_embedding_normalized = sentence_embedding / norms

                # Append embeddings for the batch to the overall list
                embeddings.extend(sentence_embedding_normalized.cpu().numpy().astype('float32'))

            return embeddings

        # Load FAISS index and metadata
        # C
        index_path = os.path.join(os.path.dirname(__file__), 'merged_index_19000.index')
        metadata_path = os.path.join(os.path.dirname(__file__), 'merged_metadata_19000.csv')

         # Load the metadata
        if not os.path.exists(metadata_path):
            raise Exception(f"Metadata file not found: {metadata_path}")
        metadata_df = pd.read_csv(metadata_path)

        # Load the FAISS index
        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
        else:
            raise Exception(f"No FAISS index file found at: {index_path}")

        # Function to query the FAISS database - UPDATED VIN'S VERSION november 21st
        def query_database(query_sentence, index_path, metadata_path, top_k=10):
            # Generate query embedding
            query_embedding = np.array(get_sentence_embedding([query_sentence])).astype('float32')

            # Perform the search
            similarities, indices = index.search(query_embedding, top_k * 3)
            distances = 1 - similarities  # Convert similarities to cosine distances

            # Filter results
            similar_sentences = []
            count = 0
            is_female_query = 'female' in query_sentence.lower()

            for i in range(len(indices[0])):
                sentence = metadata_df.iloc[indices[0][i]]['sentence']
                if is_female_query:
                    if re.search(r'\bfemale\b', sentence, re.IGNORECASE):
                        similar_sentences.append(sentence)
                        count += 1
                else:
                    if not re.search(r'\bfemale\b', sentence, re.IGNORECASE):
                        similar_sentences.append(sentence)
                        count += 1
                if count == top_k:
                    break

            return similar_sentences, distances[0]

        # Query the database
        query_sentence = f"{Sex} {Age} {Medication} {Disease}"
        index_path = os.path.join(os.path.dirname(__file__), 'merged_index_1900.index')
        metadata_path = os.path.join(os.path.dirname(__file__), 'merged_metadata_1900.csv')
        similar_sentence, distances = query_database(query_sentence, index_path=index_path, metadata_path=metadata_path)

        # Format the results and store in the dictionary
        formatted_results = "Top 10 most similar sentences with distances:\n"
        formatted_results += "Report ID\tGender\tAge\tSeriousness\tOther Medical Conditions\tHeight\tWeight\tSide Effect Name\tSystem Organ Affected\tMedication\tIndication\tSimilarity Score\n"


        for sentence, distance in zip(similar_sentence, distances):
            formatted_results += f"{sentence}\t{distance}\n"


        # Add to the dictionary under the "Relevant_ADR_reports" key
        results_dict["Relevant_ADR_reports"] = formatted_results


        #CHATGPT QUERY

        # Set the API key directly (make sure it's correct)
        openai.api_key = os.getenv('OPENAI_API_KEY')

        # Prepare your question for ChatGPT
        results_summary = (
            f"Clinical Trial Data:\n{results_dict['Clinical_trial_data']}\n\n"
            f"Relevant ADR Reports:\n{results_dict['Relevant_ADR_reports']}\n\n"
            f"ADR Statistics:\n{results_dict['ADR_statistics']}"
        )
        # Format the prompt using an f-string
        prompt = f"""
        A patient wants to know what are the risks of getting an adverse drug reaction to {Medication} taken to treat {Disease}. 
        They are a {Age}-year-old {Ethnicity} {Sex}. 
        To assess the chance that they may have an adverse reaction, please find attached the eligibility criteria for clinical trials for this drug, 
        the most relevant reported adverse drug reactions, and some statistics about the reports made for this disease. 
        Based on this information, all trials, and the adverse drug reaction reports, can you tell them what their risk 
        of having an adverse reaction to this particular drug could be in likelihood based on their race, age, and gender? 
        When not sure, please say so. When you are confident, please give likelihoods and explanation. 
        Please always format your response as follows:

        1. Clinical Trial
            - Is the patient well represented in the clinical trial for that medication for age, sex and ethnicity, consider all trials?
        
        2. Statistics about the recorded side effect
            - Look at the ADR Statistic and compare it to the patients characteristics: age and sex
        
        3. Most relevant ADR reports
            - look at the most Relevant ADR reports and compare it to the patients characteristics, age and sex
            - Go into the specifics of what kind of adverse reaction the patients exhibited for that medication
            - Consider all very relevant reports
        
        4. Risk Assessment
            - Likelihood of ADRs based on the patient's age, ethnicity, and gender
            - Any relevant caveats or uncertainties
        
        5. Recommendation
            - Final summary and any suggested actions for the patient
        """
        combined_content = f"{prompt}\n\n{results_summary}"

        # Call ChatGPT API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if you're using GPT-4
            messages=[
                {"role": "system", "content": "You are an assistant."},
                {"role": "user", "content": combined_content}
            ],
            max_tokens=2000  # Adjust as needed
        )

        # Print the response
        result = response['choices'][0]['message']['content']
        print("ChatGPT Response: \n\n", result)
        # Flatten trial_ids and remove any unwanted formatting
        cleaned_trial_ids = []
        for trial_id in trial_ids:
            if isinstance(trial_id, set):  # If the item is a set, convert it to a list and extend
                cleaned_trial_ids.extend(list(trial_id))
            else:
                cleaned_trial_ids.append(trial_id)

        # Convert all elements to strings and join without extra characters
        trial_ids_string = ", ".join(cleaned_trial_ids)

        # Print the cleaned result
        trials = (f"Relevant clinical trials analyzed:", trial_ids_string, "are accessible for consultation at https://clinicaltrials.gov/")
        source = (f"Adverse drug reaction reports have been sourced from the MedEffect Canada database: https://www.canada.ca/en/health-canada/services/drugs-health-products/medeffect-canada.html")

        return render_template('home.html', result=result, trials=trials, source=source)
    
    return render_template('home.html')

