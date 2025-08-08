from flask import Flask, request, send_file
from rembg import remove
from io import BytesIO
from PIL import Image

app = Flask(__name__)

@app.route('/remove-bg', methods=['POST'])
def remove_bg():
    if 'image' not in request.files:
        return {"error": "No image uploaded"}, 400

    input_image = request.files['image'].read()

    # Remove background using the u2netp model
    output_image = remove(input_image, model_name="u2netp")

    return send_file(BytesIO(output_image), mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
