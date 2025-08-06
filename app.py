import download_model

from flask import Flask, request, send_file
from basnet_remover import remove
import os

app = Flask(__name__)

@app.route("/")
def home():
    return "Background Remover API using BASNet is running!"

@app.route("/remove-bg", methods=["POST"])
def remove_bg():
    if 'image' not in request.files:
        return {"error": "No image file provided"}, 400

    image = request.files['image']
    input_path = "input.jpg"
    output_path = "static/output.png"

    image.save(input_path)
    remove(input_path, output_path)

    return send_file(output_path, mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True)
