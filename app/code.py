
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
        result = "Hello, World!"
        return render_template('home.html', result=result)
    
    return render_template('home.html')

