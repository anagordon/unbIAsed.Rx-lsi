from flask import Blueprint, render_template, request
import openai
import json
import pandas as pd
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch
import pandas as pd
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
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

code = Blueprint('code', __name__)

@code.route('/', methods=['GET', 'POST'])
def index():
    #Write Code Here
    result = ""

    if request.method == 'POST':
        # Variables
        Medication = request.form.get('Medication')
        Age = request.form.get('Age')
        Sex = request.form.get('Sex')
        Disease = request.form.get('Disease')
        Ethnicity = request.form.get('Ethnicity')

        # Initialize the results dictionary
        results_dict = {
            "Clinical_trial_data": "",
            "Relevant_ADR_reports": "",
            "ADR_statistics": ""
        }

        #EXTRACT THE CLINICAL TRIALS FOR A SPECIFIC MEDICATION AND DISEASE
        payload = {
            "query.cond": Disease,
            "query.term": Medication,
            "filter.overallStatus": 'COMPLETED',
            "query.intr": "Drug"
            # "pageToken":
        }
        url = "https://clinicaltrials.gov/api/v2/studies"
        # Check if the request was successful
        response = requests.get(url, params=payload)

        list(response.json().items())[0]
        json.dumps(response.json()['studies'][0])

        class DotDict(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__
            __delattr__ = dict.__delitem__

            def __init__(self, dct):
                for key, value in dct.items():
                    if hasattr(value, 'keys'):
                        value = DotDict(value)
                    self[key] = value

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
                        # Check if the required nested attributes are present
                        if (
                            hasattr(study, 'resultsSection') and
                            hasattr(study.resultsSection, 'baselineCharacteristicsModule') and
                            hasattr(study.resultsSection.baselineCharacteristicsModule, 'measures')
                        ):
                            clinical_trial_data.append(f"Study {idx + 1}:\n")
                            clinical_trial_data.append(f"{study.protocolSection.eligibilityModule}\n")
                            clinical_trial_data.append(f"{study.resultsSection.baselineCharacteristicsModule.measures}\n\n")
                    except (KeyError, AttributeError):
                        continue
                results_dict["Clinical_trial_data"] = ''.join(clinical_trial_data)  # Save clinical trial data as string

            except requests.exceptions.JSONDecodeError:
                print("Failed to decode JSON. Here is the raw response:")
                # print(response.text)
        else:
            print(f"Request failed with status code: {response.status_code}")
            # print(response.text)

        #GET THE ADR REPORT STATISTICS FOR SPECIFIC MEDICATION
        csv_file_path = os.path.join(os.path.dirname(__file__), 'ADRdata.csv')
        df = pd.read_csv(csv_file_path, delimiter='\t')

        # Define the specific string you're looking for
        specific_string = Medication.upper()

        # Filter rows where the DRUGNAME column contains the specific string
        filtered_df = df[df['DRUGNAME'].str.contains(specific_string, case=False, na=False)]

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

        # ----- GET SERIOUSNESS INFORMATION -----
        seriousness_counts = df['SERIOUSNESS_ENG'].value_counts()
        seriousness_percentages = (seriousness_counts / seriousness_counts.sum()) * 100

        serious_percentage = seriousness_percentages.get('Serious', 0)
        not_serious_percentage = seriousness_percentages.get('Not Serious', 0)
        seriousness_sentence = f"Out of all the reported ADRs, {serious_percentage:.2f}% were serious and {not_serious_percentage:.2f}% were not serious."

        # Combine all sentences into one string and save to ADR_statistics key
        results_dict["ADR_statistics"] = f"{age_group_sentence}\n{gender_sentence}\n{seriousness_sentence}"

        #GET MOST RELEVANT ADR REPORTS, will be updated later!!!
        #Load BERT model and tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', output_hidden_states=True)

        # Function to get sentence embeddings
        def get_sentence_embedding(sentence):
            tokens = tokenizer.tokenize(sentence)
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            T = len(tokens)
            attention_mask = [1] * T
            seg_ids = [0] * T
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            token_ids = torch.tensor(input_ids).unsqueeze(0)
            attention_mask = torch.tensor(attention_mask).unsqueeze(0)
            seg_ids = torch.tensor(seg_ids).unsqueeze(0)
            outputs = model.bert(token_ids, attention_mask=attention_mask, token_type_ids=seg_ids)
            sentence_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
            return sentence_embedding

                
        # Load FAISS index and metadata
        index_file_path = os.path.join(os.path.dirname(__file__), 'vector_database.index')
        index = faiss.read_index(index_file_path)

        metadata_file_path = os.path.join(os.path.dirname(__file__), 'metadata.csv')
        metadata_df = pd.read_csv(metadata_file_path)

        # Function to query the FAISS database
        def query_database(query_sentence, top_k=10):
            query_embedding = np.array(get_sentence_embedding(query_sentence)[0]).astype('float32').reshape(1, -1)
            distances, indices = index.search(query_embedding, top_k * 2)
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

            return similar_sentences, distances[0][:len(similar_sentences)]

        # Query the database
        query_sentence = f"{Sex} {Age}"
        similar_sentences, distances = query_database(query_sentence)

        # Format the results and store in the dictionary
        formatted_results = "Top 10 most similar sentences with distances:\n"
        for sentence, distance in zip(similar_sentences, distances):
            formatted_results += f"{sentence}\t{distance}\n"

        # Add to the dictionary under the "Relevant_ADR_reports" key
        results_dict["Relevant_ADR_reports"] = formatted_results

        #CHATGPT QUERY
        # Set the API key directly (make sure it's correct)
        openai.api_key = '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'

        # Prepare your question for ChatGPT
        results_summary = (
            f"Clinical Trial Data:\n{results_dict['Clinical_trial_data']}\n\n"
            f"Relevant ADR Reports:\n{results_dict['Relevant_ADR_reports']}\n\n"
            f"ADR Statistics:\n{results_dict['ADR_statistics']}"
        )
        # Format the prompt using an f-string
        prompt = (
            f"A patient wants to know what are the risks of getting an adverse drug reaction to {Medication} taken to treat {Disease}. "
            f"They are a {Age}-year-old {Ethnicity} {Sex}. "
            "To assess the chance that they may have an adverse reaction, please find attached the eligibility criteria for clinical trials for this drug, "
            "the most relevant reported adverse drug reactions, and some statistics about the reports made for this disease. "
            "Based on this information, all trials, and the adverse drug reaction reports, can you tell them what their risk "
            "of having an adverse reaction to this particular drug could be in likelihood based on their race, age, and gender? "
            "When not sure, please say so. When you are confident, please give likelihoods and explanation."
        )

        combined_content = f"{prompt}\n\n{results_summary}"

        # Call ChatGPT API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if you're using GPT-4
            messages=[
                {"role": "system", "content": "You are an assistant."},
                {"role": "user", "content": combined_content}
            ],
            max_tokens=500  # Adjust as needed
        )

        # Print the response
        answer = response['choices'][0]['message']['content']
        # print("ChatGPT Response:", answer)

        result = answer
        return render_template('home.html', result=result)
    
    return render_template('home.html')

