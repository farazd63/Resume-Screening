from flask import Flask, render_template, request
import pickle
import os
import pdfplumber
import docx
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load model & vectorizer
model = pickle.load(open(r'C:\Users\This PC\OneDrive\OneDrive - Islamabad Model Postgraduate College of Commerce H-8 4 Islamabad\SZABIST WORKING ZONE\Project\myproject\models\rf_classifier.pkl', 'rb'))
vectorizer = pickle.load(open(r'C:\Users\This PC\OneDrive\OneDrive - Islamabad Model Postgraduate College of Commerce H-8 4 Islamabad\SZABIST WORKING ZONE\Project\myproject\models\tfidf_vectorizer.pkl', 'rb'))

# Extract text from PDF
def extract_text_pdf(filepath):
    with pdfplumber.open(filepath) as pdf:
        return " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])

# Extract text from DOCX
def extract_text_docx(filepath):
    doc = docx.Document(filepath)
    return " ".join([para.text for para in doc.paragraphs])

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    error_msg = None

    if request.method == 'POST':
        file = request.files.get('resume')
        if not file:
            error_msg = "Please upload a resume file."
        else:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            if filename.endswith('.pdf'):
                text = extract_text_pdf(filepath)
            elif filename.endswith('.docx'):
                text = extract_text_docx(filepath)
            else:
                error_msg = "Unsupported file format. Please upload a PDF or DOCX file."
                return render_template("index.html", error=error_msg)

            if text.strip() == "":
                error_msg = "Failed to extract content from resume."
            else:
                vector = vectorizer.transform([text])
                probabilities = model.predict_proba(vector)[0]
                categories = model.classes_
                top_idx = np.argsort(probabilities)[::-1]
                prediction = [(categories[i], round(probabilities[i] * 100, 2)) for i in top_idx[:3]]

    return render_template('index.html', prediction=prediction, error=error_msg)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
