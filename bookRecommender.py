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
            <link rel="stylesheet" type="text/css" href="static/style.css">
        </head>
        <body>
            <h1>Welcome to the Book Recommendation System!</h1>
            <form method="POST" action="/recommend">
                <input type="text" name="book_title" placeholder="Enter a book title for a book that you like" required>
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
            <h2>Recommended Books:</h2>
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
        recommendations_html += '</table>'

        return render_template_string(recommendations_html)

# Start Flask app
if __name__ == '__main__':
    app.run(debug=True)