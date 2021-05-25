import os
from flask import Flask, render_template, request

#from preprocessing import preprocessing_main
from query import Query

"""
This is the main file to run the medical information retrieval server.
The following dataset can be used to retrieve similar images: https://publications.rwth-aachen.de/record/667228
"""

# database_path = "static/images/database/"

feedback_result = None
selected_image = None
app = Flask(__name__)

# query = Query(path_to_index= "static/index.csv")

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

    return render_template("start.html", selected_image=f.filename)


@app.route("/query_result", methods=['POST'])
def start_query():

    # TODO:
    ret_img_pathes = []

    query = Query(query_image_name=app.config['IMAGE_DB'] + os.sep + selected_image)
    query_result = query.run()
    correct_prediction_dictionary = query.check_code(query_result)
    for tup in query_result:
        ret_img_pathes.append(app.config['IMAGE_DB'] + os.sep + tup[1].split(os.sep)[-1])

    print("Retrieved images: ", query_result)
    print("correct_prediction_dictionary:")
    print(correct_prediction_dictionary)



    return visualize_query(ret_img_pathes)  # vorher übergeben query_results

def visualize_query(query_result):
    # TODO:

    # return render_template("query_result.html",
    #    zipped_input=zip([selected_image], input_code, input_info),
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