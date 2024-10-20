from flask import Blueprint, render_template, request
import openai
import requests
from bs4 import BeautifulSoup

# Set the API key directly (make sure it's correct)
openai.api_key = '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'

code = Blueprint('code', __name__)

# Function to extract content from a website
def get_website_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract the text from the website (you can customize this as needed)
        print('WHAT WAS EXTRACTED' + soup.get_text())
        return soup.get_text()
    except Exception as e:
        return f"Error fetching website content: {str(e)}"

@code.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    
    if request.method == 'POST':
        prompt = request.form.get('prompt')
        url = request.form.get('url')  # Get the website URL from the form

        # Fetch website content
        website_content = get_website_content(url)

        try:
            # Send the website content along with the prompt to the ChatCompletion API
            completion = openai.ChatCompletion.create(
                model="gpt-4",  # Ensure you use a proper model here
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Here is the content of the website: {website_content} \n\n Now, {prompt}"}
                ],
                max_tokens=1000
            )
            result = completion['choices'][0]['message']['content']
        
        except Exception as e:
            result = f"Error: {str(e)}"
    
    return render_template('home.html', result=result)
