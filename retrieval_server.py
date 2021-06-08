import os
from os.path import join
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename

from query import Query
from preprocessing import build_index
from irma_code import get_img_info


"""
This is the main file to run the medical information retrieval server.
The following dataset can be used to retrieve similar images: https://publications.rwth-aachen.de/record/667228
"""

# database_path = "static/images/database/"

feedback_result = None
selected_image = None
selected_image_path = None
query = None


app = Flask(__name__)

# query = Query(path_to_index= "static/codes/index.csv")

elements_per_page = 10
page= 1

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = join('static', 'uploads')
# app.config['UPLOAD_FOLDER'] = 'static' + os.sep + 'uploads'
app.config['IMAGE_DB'] = join('static', 'img_db')
# app.config['IMAGE_DB'] = 'static' + os.sep + 'img_db'


@app.route("/")
def index():
    global selected_image
    return render_template("start.html", selected_image=selected_image)


@app.route("/selected_image", methods=['POST'])
def select_query_image():
    # TODO:
    if request.method == 'POST':
        f = request.files['file']
        new_path = join(app.config['UPLOAD_FOLDER'],  f.filename)
        f.save(new_path)

        global selected_image
        selected_image = f.filename

    return render_template("start.html", selected_image=selected_image)


@app.route("/query_result", methods=['POST'])
def start_query():
    ret_img_pathes = []

    # TODO:
    #ret_img_pathes = []
    global query
    query = Query(query_image_name= join(app.config['IMAGE_DB'], selected_image))
    query_result = query.run()
    correct_prediction_dictionary = query.check_code(query_result)

    print("Retrieved images: ", query_result)
    print("correct_prediction_dictionary:")
    print(correct_prediction_dictionary)

    #return visualize_query(ret_img_pathes)  # vorher übergeben query_results
    return visualize_query(query_result)

    #return visualize_query(ret_img_pathes)

def visualize_query(query_result):

    ret_img_names = []
    ret_img_pathes = []
    ret_img_distances = []

    for (distance, img_path) in query_result:
        ret_img_names.append(img_path.split(os.sep)[-1])
        ret_img_pathes.append(app.config['IMAGE_DB'] + os.sep + img_path.split(os.sep)[-1])
        ret_img_distances.append(round(distance, 2))

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
    global feedback_result

    # POST request
    if request.method == 'POST':

        # TODO:
        print(request.is_json)
        print(request.get_json())

        relevant = request.get_json()[0]
        non_relevant = request.get_json()[1]

        relevant = [e.replace(".png", "") for e in relevant]
        non_relevant = [e.replace(".png", "") for e in non_relevant]


        global query
        #query = Query(query_image_name=app.config['IMAGE_DB'] + os.sep + selected_image)
        feedback_result = query.relevance_feedback(relevant, non_relevant)
        correct_prediction_dictionary = query.check_code(feedback_result)
        print("Retrieved images: ", feedback_result)

        return redirect('/relevance_feedback')
        # return visualize_query(feedback_result)

    if request.method == 'GET':
        print('here')
        return visualize_query(feedback_result)



if __name__ == "__main__":
    app.run(port=4555, debug=True)