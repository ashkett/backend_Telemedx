from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as palm
import pdfplumber
import re  # Regular expressions for splitting columns
import os
import pandas as pd
import PyPDF2

app = Flask(__name__)
CORS(app)

key = "AIzaSyDxByW2GpWT0OKSLifZbiFatBZyG_8QfDE"
if not key:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

palm.configure(api_key=key)
model = palm.GenerativeModel("gemini-1.5-flash")

@app.route('/summarize', methods=['POST'])
def summarize():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded_file = request.files['file']
    extracted_text = ""

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"

    if not extracted_text:
        return jsonify({"error": "No text extracted from the PDF"}), 400
    print(extracted_text)

    try:
        # Create a prompt to generate a summary
        prompt = f"""
        Please provide a concise and informative summary of the following text, focusing on the key points such as the name of the medicine, its dosage, timing, frequency, and duration. Aim for a summary that is about 200-250 words long, capturing the essential details. Maintain the original tone and avoid adding any personal opinions or interpretations.

        The details are as follows:

        ```
        {extracted_text}
        ```

        Summarize the medication details including their use, dosage instructions, and duration in a clear and simple manner.
        """

        # Send the prompt to the AI model for summary generation
        response = model.generate_content(prompt)
        summary = response.text

        if not summary:
            return jsonify({"error": "No summary generated"}), 500

        return jsonify({"summary": summary})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    




# Load dataset from CSV
csv_file_path = "./Medicine_Details.csv"
data = pd.read_csv(csv_file_path)

# Ensure necessary columns exist
required_columns = ["Medicine Name", "Composition", "Excellent Review %", "Average Review %", "Poor Review %"]
if not all(col in data.columns for col in required_columns):
    raise ValueError("CSV file must contain these columns: " + ", ".join(required_columns))

def extract_ingredients(comp):
    """Extracts active ingredients while keeping their dosage information."""
    return set(ingredient.strip().lower() for ingredient in comp.split("+"))

def composition_match(comp1, comp2):
    """Calculates the percentage of matching ingredients between two compositions."""
    ingredients1 = extract_ingredients(comp1)
    ingredients2 = extract_ingredients(comp2)

    if not ingredients1 or not ingredients2:
        return 0

    match_count = len(ingredients1.intersection(ingredients2))
    
    return (match_count / len(ingredients1)) * 100  # Normalize to percentage

def get_alternative_medicines(med_name, top_n=3):
    """Returns alternative medicines based on composition match and review score."""
    if med_name not in data["Medicine Name"].values:
        return []

    med_idx = data[data["Medicine Name"] == med_name].index[0]
    target_composition = data.loc[med_idx, "Composition"]
    data["composition_match"] = data["Composition"].apply(lambda x: composition_match(target_composition, x))
    data["review_score"] = data["Excellent Review %"] + data["Average Review %"] - data["Poor Review %"]
    data["final_score"] = (0.7 * data["composition_match"]) + (0.3 * data["review_score"])

    results = data[["Medicine Name", "composition_match", "final_score"]].copy()
    results = results[results["Medicine Name"] != med_name].sort_values(by=["final_score"], ascending=False)

    return results.head(top_n).to_dict(orient="records")

def extract_medicine_names_from_pdf(pdf_path):
    """Extracts text from PDF and finds medicine names from dataset."""
    medicine_names = set(data["Medicine Name"].str.lower())  # Lowercase for matching
    extracted_medicines = set()

    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                words = text.lower().split()
                extracted_medicines.update(medicine_names.intersection(words))

    return list(extracted_medicines)

@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    pdf_file = request.files["file"]

    if pdf_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not pdf_file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    # Save the uploaded file temporarily
    temp_path = "temp_uploaded.pdf"
    pdf_file.save(temp_path)

    try:
        # Extract medicine names from PDF
        extracted_medicines = extract_medicine_names_from_pdf(temp_path)

        # Get alternatives for each extracted medicine
        alternatives = {med: get_alternative_medicines(med) for med in extracted_medicines}

        return jsonify(alternatives)

    finally:
        # Remove temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)



if __name__ == '__main__':
    app.run(debug=True)
