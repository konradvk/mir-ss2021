from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = 'static'

'''
Tutorial help: https://www.tutorialspoint.com/flask/flask_file_uploading.htm
'''

@app.route('/')
def start():
   return render_template('upload.html')


@app.route("/selected_image", methods=['POST'])
def select_query_image():
   if request.method == 'POST':
      # Save image to static folder
      f = request.files['file']
      new_path = os.path.join(
               app.config['UPLOAD_FOLDER'], 
               secure_filename(f.filename))
      f.save(new_path)

      # Display image
      return render_template('show_image.html', image = new_path)


if __name__ == '__main__':
   app.run(debug = True)