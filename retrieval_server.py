import os
from os.path import join
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename

from query import Query
from preprocessing import build_index
from irma_code import get_img_info
import evaluation as ev


"""
This is the main file to run the medical information retrieval server.
The following dataset can be used to retrieve similar images: https://publications.rwth-aachen.de/record/667228
"""

feedback_result = None
selected_image = None
query = None


app = Flask(__name__)

page= 1

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = join('static', 'uploads')
app.config['IMAGE_DB'] = join('static', 'img_db')


@app.route("/")
def index():
    global selected_image
    return render_template("start.html", selected_image=selected_image)

@app.route("/clear")
def clear():
    global selected_image
    selected_image = None
    return redirect("/")

@app.route("/selected_image", methods=['POST'])
def select_query_image():
    if request.method == 'POST':
        f = request.files['file']
        new_path = join(app.config['UPLOAD_FOLDER'],  f.filename)
        f.save(new_path)

        global selected_image
        selected_image = f.filename

    return render_template("start.html", selected_image=selected_image)


@app.route("/query_result", methods=['POST'])
def start_query():

    global selected_image
    global query
    query = Query(query_image_name= join(app.config['IMAGE_DB'], selected_image))
    query_result = query.run()
    correct_prediction_dictionary = query.check_code(query_result)

    print("\nQuery result:\n", query_result)
    print("\nCorrect predictions:\n", correct_prediction_dictionary, "\n")

    correct_prediction_ls = list(correct_prediction_dictionary.values())
    print("P@K: ", ev.precision_at_k(correct_prediction_ls))

    return visualize_query(query_result)



def visualize_query(query_result):

    global selected_image

    # Get all information connected to input image
    input_irma = get_img_info( [selected_image] )
    input_info =    [   join(app.config['IMAGE_DB'], selected_image),
                        input_irma[0]
                    ]

    # Get all information connected to the query result (qr)
    qr_names, qr_distances, qr_paths = [],[],[]
    for (distance, img_path) in query_result:
        img_name = img_path.split(os.sep)[-1]
        qr_names.append(img_name)
        qr_paths.append( join(app.config['IMAGE_DB'], img_name))
        qr_distances.append(round(distance, 2))

    qr_irma = get_img_info(qr_names)
    qr_info = []
    for i in range(len(qr_names)):
        qr_info.append([qr_paths[i], qr_distances[i], qr_irma[i]])
    
    return render_template('query_result.html', input_info=input_info, query_result_info=qr_info)


@app.route("/recalc", methods=['POST'])
def recalc_index():

    build_index()

    return render_template("start.html", selected_image=selected_image)

@app.route("/new_page", methods=['POST'])
def new_page():
    return start_query()


@app.route('/relevance_feedback', methods=['POST', 'GET'])
def relevance_feedback():
    global feedback_result

    # POST request
    if request.method == 'POST':

        # print(request.is_json)
        # print(request.get_json())

        relevant = request.get_json()[0]
        non_relevant = request.get_json()[1]

        relevant = [e.replace(".png", "") for e in relevant]
        non_relevant = [e.replace(".png", "") for e in non_relevant]


        global query
        feedback_result = query.relevance_feedback(relevant, non_relevant)
        correct_prediction_dictionary = query.check_code(feedback_result)

        print("\nRelevance Feedback result:\n", feedback_result)
        print("\nCorrect predictions:\n", correct_prediction_dictionary, "\n")
        correct_prediction_ls = list(correct_prediction_dictionary.values())
        print("P@K: ", ev.precision_at_k(correct_prediction_ls))

        return redirect('/relevance_feedback')

    if request.method == 'GET':
        return visualize_query(feedback_result)



if __name__ == "__main__":
    app.run(port=4555, debug=True)