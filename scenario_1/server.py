from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from PIL import Image







app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

'''
Tutorial help: https://www.tutorialspoint.com/flask/flask_file_uploading.htm
'''

@app.route('/')
def start():
   return render_template('upload.html')
	
@app.route("/selected_image", methods=['POST'])
def select_query_image():
    f = request.files['file']
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
    return render_template('show_image.html', uploaded_image=secure_filename(f.filename))
		
if __name__ == '__main__':
   app.run(debug = True)