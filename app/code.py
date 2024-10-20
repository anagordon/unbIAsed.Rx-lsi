from flask import Blueprint, render_template, request
import openai

# Set the API key directly (make sure it's correct)
openai.api_key = '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'

code = Blueprint('code', __name__)

@code.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    
    if request.method == 'POST':
        prompt = request.form.get('prompt')

        try:
            completion = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            result = completion.choices[0].message
        
        except Exception as e:
            result = f"Error: {str(e)}"
    
    return render_template('home.html', result=result)