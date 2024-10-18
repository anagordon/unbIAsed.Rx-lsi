
from flask import Blueprint, render_template, request

code = Blueprint('code', __name__)

@code.route('/', methods=['GET', 'POST'])
def index():
    #Write Code Here
    result = ""

    if request.method == 'POST':
        #Write Code Here
        result = "Hello, World!"
        return render_template('home.html', result=result)
    
    return render_template('home.html')
