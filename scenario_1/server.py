from flask import Flask, render_template, request, url_for
import os
# from werkzeug.utils import secure_filename
from pathlib import Path
from query import Query

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = 'static'

app.config['UPLOAD_FOLDER'] = 'static' + os.sep + 'uploads'
app.config['IMAGE_DB'] = 'static' + os.sep + 'img_db'

'''
Tutorial help: https://www.tutorialspoint.com/flask/flask_file_uploading.html
'''


@app.route('/')
def start():
    return render_template('upload.html')


@app.route("/selected_image", methods=['POST'])
def select_query_image():

    ret_img_pathes = []

    if request.method == 'POST':
        f = request.files['file']
        f.save(app.config['UPLOAD_FOLDER'] + os.sep + f.filename)

        query = Query(query_image_name=app.config['IMAGE_DB'] + os.sep + f.filename)
        query_result = query.run()
        correct_prediction_dictionary = query.check_code(query_result)
        for tup in query_result:
            ret_img_pathes.append(app.config['IMAGE_DB'] + os.sep + tup[1].split(os.sep)[-1])

        print("Retrieved images: ", query_result)
        print("correct_prediction_dictionary:")
        print(correct_prediction_dictionary)

    return render_template('show_image.html', ret_img_ls=ret_img_pathes)  # img=url_for('static', filename=f.filename)


if __name__ == '__main__':
    app.run(debug=True)