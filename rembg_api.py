from flask import Flask, request, send_file
from flask_cors import CORS  # ✅ important
from rembg import remove
from io import BytesIO

app = Flask(__name__)

# ✅ Correct and complete CORS setup
CORS(app)

@app.route('/remove-bg', methods=['POST'])
def remove_bg():
    print("✅/remove-bg route hit")
    print("request.files:", request.files)

    if 'image_file' not in request.files:
        print("❌ image_file not found")
        return "Missing image_file", 400

    file = request.files['image_file']
    input_image = file.read()

    try:
        output_image = remove(input_image)
        print("✅ Background removed")
        return send_file(BytesIO(output_image), mimetype='image/png')
    except Exception as e:
        print("❌ Background removal failed:", e)
        return "Error removing background", 500

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5050)

