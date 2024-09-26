import os
from flask import Flask, request, render_template, jsonify
import PyPDF2
import docx2txt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['EXPERT_FOLDER'] = 'expert_portfolios/'




# Helper Functions for Text Extraction
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text


def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)


def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    if file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    else:
        return ""


# Load expert portfolios
def load_expert_profiles():
    expert_profiles = []
    expert_files = os.listdir(app.config['EXPERT_FOLDER'])
    for file_name in expert_files:
        file_path = os.path.join(app.config['EXPERT_FOLDER'], file_name)
        expert_profiles.append({
            'name': file_name,
            'resume_text': extract_text(file_path)
        })
    return expert_profiles


# Flask route to handle the main page and display form
@app.route("/")
def index():
    return render_template('index.html')


# Route for matching and ranking applicant resume against experts
@app.route("/matcher", methods=['POST'])
def matcher():
    # Get the text input and file input from the form
    job_description = request.form.get('job_description')
    applicant_file = request.files.get('applicant_resume')

    if not job_description or not applicant_file:
        return jsonify({'error': 'Job description and resume are required.'})

    # Save applicant resume
    applicant_path = os.path.join(app.config['UPLOAD_FOLDER'], applicant_file.filename)
    applicant_file.save(applicant_path)

    # Extract text from applicant resume
    applicant_text = extract_text(applicant_path)

    # Load expert resumes
    expert_profiles = load_expert_profiles()

    if not expert_profiles:
        return jsonify({'error': 'No expert profiles found.'})

    # Extract text from all expert resumes
    expert_texts = [expert['resume_text'] for expert in expert_profiles]
    expert_names = [expert['name'] for expert in expert_profiles]

    # Vectorization and Cosine Similarity
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([applicant_text] + expert_texts)
    applicant_vector = vectors[0]
    expert_vectors = vectors[1:]

    # Calculate cosine similarity
    similarities = cosine_similarity(applicant_vector, expert_vectors)[0]

    # Rank the experts based on similarity
    ranked_indices = similarities.argsort()[::-1]
    top_5_experts = [(expert_names[i], similarities[i]) for i in ranked_indices[:5]]

    # Return the top 5 experts as a JSON response
    return jsonify({
        'status': 'success',
        'top_5_experts': [
            {'name': expert[0], 'similarity_score': round(expert[1], 4)}
            for expert in top_5_experts
        ]
    })


# Flask route to handle multiple expert portfolio uploads
@app.route('/upload_experts', methods=['POST'])
def upload_experts():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No file part'})

    files = request.files.getlist('files[]')

    if len(files) == 0:
        return jsonify({'error': 'No selected files'})

    saved_files = []
    for file in files:
        if file.filename:
            file_path = os.path.join(app.config['EXPERT_FOLDER'], file.filename)
            file.save(file_path)
            saved_files.append(file.filename)

    return jsonify({'status': 'success', 'message': f'{len(saved_files)} expert portfolios uploaded successfully'})


if __name__ == "__main__":
    app.run(debug=True, port=5001)  # or any other available port
