import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template_string

# Create Flask app
app = Flask(__name__)

# Read the pre-processed csv-file
df = pd.read_csv("data/Books_Preprocessed.csv")
df = df[["ISBN", "Title", "Author", "Year", "Publisher", "Processed_Title"]].dropna()
df.columns = ["ISBN", "Title", "Author", "Year", "Publisher", "Processed_Title"]
df = df.drop_duplicates(subset=["Title"])

# Load Spacy model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]  # Lemmatize text
    return " ".join(tokens)

#Keep commented, as the dataset has already been preprocessed
#df["Processed_Title"] = df["Title"].apply(preprocess_text)

# Create TfidfVectorizer & TF-IDF matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["Processed_Title"])

# Calculate cosine similarities
cosine_sim = cosine_similarity(tfidf_matrix, dense_output=False)

def recommend_books(book_title, num_recommendations=15):
    processed_input = preprocess_text(book_title)
    input_vector = vectorizer.transform([processed_input])
    
    similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()
    
    similar_indices = similarities.argsort()[::-1][1:num_recommendations+1]
    
    recommended_books = df.iloc[similar_indices][["Title", "Author", "Year"]]
    return recommended_books

# Flask route
@app.route('/')
def index():
    return '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Book Recommendation System</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f8f9fa;
                }
                h1 {
                    color: #2c3e50;
                    text-align: center;
                    margin-bottom: 30px;
                    padding-bottom: 15px;
                    border-bottom: 2px solid #e74c3c;
                }
                form {
                    display: flex;
                    flex-direction: column;
                    max-width: 600px;
                    margin: 0 auto;
                    background: white;
                    padding: 25px;
                    border-radius: 8px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }
                input[type="text"] {
                    padding: 12px;
                    margin-bottom: 15px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    font-size: 16px;
                }
                input[type="submit"] {
                    background-color: #e74c3c;
                    color: white;
                    border: none;
                    padding: 12px;
                    cursor: pointer;
                    border-radius: 4px;
                    font-size: 16px;
                    font-weight: bold;
                    transition: background-color 0.3s;
                }
                input[type="submit"]:hover {
                    background-color: #c0392b;
                }
                #recommendations {
                    margin-top: 30px;
                }
            </style>
        </head>
        <body>
            <h1>Welcome to the Book Recommendation System!</h1>
            <form method="POST" action="/recommend">
                <input type="text" name="book_title" placeholder="Enter a book title that you like" required>
                <input type="submit" value="Get Recommendations">
            </form>
            <div id="recommendations"></div>
        </body>
        </html>
    '''

# Flask route recommend endpoint
@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form['book_title']
    if user_input:
        recommendations = recommend_books(user_input)
        recommendations_html = '''
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Book Recommendations</title>
                <style>
                    body {
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                        background-color: #f8f9fa;
                    }
                    h2 {
                        color: #2c3e50;
                        text-align: center;
                        margin-bottom: 25px;
                        padding-bottom: 10px;
                        border-bottom: 2px solid #e74c3c;
                    }
                    table {
                        width: 100%;
                        border-collapse: collapse;
                        margin: 25px 0;
                        background-color: white;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                        border-radius: 8px;
                        overflow: hidden;
                    }
                    th, td {
                        padding: 15px;
                        text-align: left;
                        border-bottom: 1px solid #ddd;
                    }
                    th {
                        background-color: #e74c3c;
                        color: white;
                        font-weight: bold;
                    }
                    tr:hover {
                        background-color: #f5f5f5;
                    }
                    tr:nth-child(even) {
                        background-color: #f9f9f9;
                    }
                    .back-button {
                        display: block;
                        width: 200px;
                        margin: 20px auto;
                        padding: 10px;
                        background-color: #3498db;
                        color: white;
                        text-align: center;
                        text-decoration: none;
                        border-radius: 4px;
                        font-weight: bold;
                    }
                    .back-button:hover {
                        background-color: #2980b9;
                    }
                </style>
            </head>
            <body>
                <h2>Recommended Books Based on Your Selection</h2>
                <table>
                    <tr>
                        <th>Title</th>
                        <th>Author</th>
                        <th>Year</th>
                    </tr>
        '''
        for index, row in recommendations.iterrows():
            recommendations_html += f'''
                <tr>
                    <td>{row['Title']}</td>
                    <td>{row['Author']}</td>
                    <td>{row['Year']}</td>
                </tr>
            '''
        recommendations_html += '''
                </table>
                <a href="/" class="back-button">Search Again</a>
            </body>
            </html>
        '''

        return recommendations_html

# Start Flask app
if __name__ == '__main__':
    app.run(debug=True)