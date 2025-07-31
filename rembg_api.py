import os
from flask import Flask, request, send_file
from flask_cors import CORS
from rembg import remove
from io import BytesIO

app = Flask(__name__)
CORS(app)

@app.route('/remove-bg', methods=['POST'])
def remove_bg():
    print("✅/remove-bg route hit")
    if 'image_file' not in request.files:
        return "Missing image_file", 400

    file = request.files['image_file']
    input_image = file.read()

    try:
        output_image = remove(input_image)
        return send_file(BytesIO(output_image), mimetype='image/png')
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5050))  # fallback for local dev
    app.run(host='0.0.0.0', port=port)
