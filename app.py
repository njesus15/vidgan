import base64

from flask import Flask, request, render_template, session, redirect, url_for, flash
from model.utils import *
from werkzeug.utils import secure_filename
import os

currentDirectory = os.getcwd()
app = Flask(__name__)

app.secret_key = 'gobears'
app.config['SESSION_TYPE'] = 'filesystem'
IMAGES = os.path.join('static', 'images')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = IMAGES
app.config['MODEL'] = get_model()
app.config['HAS_OUTPUT'] = False


@app.route('/')
def login():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'homepage_header.jpg')
    return render_template('home.html', user_image=full_filename)


@app.route('/demo', methods=['GET', 'POST'])
def demo():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file: # Display, image, reduced image, and gif
            # Change input boolean
            app.config['HAS_OUTPUT'] = True
            filename = secure_filename(file.filename)
            fullname = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(fullname)
            reduced_image, output = model_output(app.config['MODEL'], fullname)

            # Save reduced image
            small_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'small_' + filename)
            reduced_image.save(small_img_path)

            # Save gif
            gif_path = fullname[:-4] + '.gif'
            encoded_string = base64.b64encode(denorm(output.data.cpu()[0]).numpy())
            app.config['json_output'] = {'gif': str(encoded_string)}
            make_gif(denorm(output.data.cpu()[0]), gif_path)
            return render_template('demo.html',
                                   image=fullname,
                                   small_image=small_img_path,
                                   gif=gif_path)

    return render_template('demo.html')

@app.route('/demo/output', methods=['GET', 'POST'])
def get_output():
    if request.method == 'GET':
        if app.config['HAS_OUTPUT']:
            return app.config['json_output']

if __name__ == "__main__":
    currentDirectory = os.getcwd()
    app.run(debug=True)
