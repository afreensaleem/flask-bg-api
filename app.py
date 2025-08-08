from flask import Flask, request, send_file
from rembg import remove
from io import BytesIO
from PIL import Image

app = Flask(__name__)

@app.route("/")
def home():
    return "Background Remover API is running!"

@app.route("/remove-bg", methods=["POST"])
def remove_bg():
    if 'image' not in request.files:
        return {"error": "No image file provided"}, 400

    image_file = request.files['image']
    input_image = Image.open(image_file.stream).convert("RGBA")
    
    output_image = remove(input_image)

    img_io = BytesIO()
    output_image.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
