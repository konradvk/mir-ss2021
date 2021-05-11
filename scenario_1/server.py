from flask import Flask, render_template, request
import os
app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

'''
Tutorial help: https://www.tutorialspoint.com/flask/flask_file_uploading.htm
'''

@app.route('/')
def start():
   return render_template('upload.html')
	
@app.route("/selected_image", methods=['POST'])
def select_query_image():
    pass
		
if __name__ == '__main__':
   app.run(debug = True)