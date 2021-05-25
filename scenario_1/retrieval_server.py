import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

#from preprocessing import preprocessing_main
from query import Query

"""
This is the main file to run the medical information retrieval server.
The following dataset can be used to retrieve similar images: https://publications.rwth-aachen.de/record/667228
"""

database_path = "static/images/database/"

feeback_result = None
selected_image = None
selected_image_path = None

app = Flask(__name__)


elements_per_page = 10
page= 1

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


app.config['QUERY_FOLDER'] = 'static' + os.sep + 'images' + os.sep + 'query'
app.config['IMAGE_DB'] = 'static' + os.sep + 'images' + os.sep + 'database'

@app.route("/")
def index():
    global selected_image
    return render_template("start.html", selected_image= selected_image)

@app.route("/selected_image", methods=['POST'])
def select_query_image():
    f = request.files['file']
    global selected_image_path
    selected_image_path = os.path.join(app.config['QUERY_FOLDER'], secure_filename(f.filename))
    f.save(selected_image_path)
    global selected_image
    selected_image = secure_filename(f.filename)

    return render_template("start.html", selected_image=selected_image)

@app.route("/query_result", methods=['POST'])
def start_query():
    ret_img_pathes = []

    query = Query(query_image_name=app.config['IMAGE_DB'] + os.sep + selected_image)
    query_result = query.run()
    correct_prediction_dictionary = query.check_code(query_result)
    for tup in query_result:
        ret_img_pathes.append(app.config['IMAGE_DB'] + os.sep + tup[1].split(os.sep)[-1])

    print("Retrieved images: ", query_result)
    print("correct_prediction_dictionary:")
    print(correct_prediction_dictionary)

    return visualize_query(ret_img_pathes)

def visualize_query(query_result):

   # return render_template("query_result.html",
   #     zipped_input=zip([selected_image], input_code, input_info),
   #  zipped_results= zip(image_names, image_distances, image_codes, irma_infos))
    return render_template('show_image.html', ret_img_ls=query_result)

@app.route("/recalc", methods=['POST'])
def recalc_index():

    # TODO:

    return render_template("start.html", selected_image= selected_image)

@app.route("/new_page", methods=['POST'])
def new_page():
    # TODO:

    return start_query()

@app.route('/relevance_feedback', methods=['POST', 'GET'])
def relevance_feedback():
    global feeback_result

    # POST request
    if request.method == 'POST':

        # TODO:
        pass


    if request.method == 'GET':
        return visualize_query(feeback_result)

if __name__ == "__main__":
    app.run(port=4555, debug=True)