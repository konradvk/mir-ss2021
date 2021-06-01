import os
from flask import Flask, render_template, request

from query import Query
from preprocessing import build_index
from irma_code_exercise import get_img_info

"""
This is the main file to run the medical information retrieval server.
The following dataset can be used to retrieve similar images: https://publications.rwth-aachen.de/record/667228
"""

# database_path = "static/images/database/"

feedback_result = None
selected_image = None
app = Flask(__name__)

# query = Query(path_to_index= "static/codes/index.csv")

elements_per_page = 10
page= 1

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = 'static' + os.sep + 'uploads'
app.config['IMAGE_DB'] = 'static' + os.sep + 'img_db'


@app.route("/")
def index():
    global selected_image
    return render_template("start.html", selected_image= selected_image)


@app.route("/selected_image", methods=['POST'])
def select_query_image():
    # TODO:
    if request.method == 'POST':
        f = request.files['file']
        new_path = app.config['UPLOAD_FOLDER'] + os.sep + f.filename
        f.save(new_path)

        global selected_image
        selected_image = f.filename

    return render_template("start.html", selected_image=selected_image)


@app.route("/query_result", methods=['POST'])
def start_query():

    # TODO:
    #ret_img_pathes = []

    query = Query(query_image_name=app.config['IMAGE_DB'] + os.sep + selected_image)
    query_result = query.run()
    correct_prediction_dictionary = query.check_code(query_result)

    print("Retrieved images: ", query_result)
    print("correct_prediction_dictionary:")
    print(correct_prediction_dictionary)

    #return visualize_query(ret_img_pathes)  # vorher übergeben query_results
    return visualize_query(query_result)

def visualize_query(query_result):

    ret_img_names = []
    ret_img_pathes = []
    ret_img_distances = []

    for (distance, img_path) in query_result:
        ret_img_names.append(img_path.split(os.sep)[-1])
        ret_img_pathes.append(app.config['IMAGE_DB'] + os.sep + img_path.split(os.sep)[-1])
        ret_img_distances.append(distance)

    ret_img_info = get_img_info(ret_img_names)

    ret_img_and_info = []

    for i in range(len(ret_img_names)):
        ret_img_and_info.append([ret_img_pathes[i], ret_img_distances[i], ret_img_info[i]])


    return render_template('query_result.html', img_infos=ret_img_and_info)

@app.route("/recalc", methods=['POST'])
def recalc_index():

    # TODO:
    build_index()


    return render_template("start.html", selected_image=selected_image)

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