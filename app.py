from flask_cors import CORS
from flask import Flask, render_template, request, jsonify, send_file, url_for
from Database.query import search_embeddings, generate_rag_response, chroma_client, collection
import os
import json
from datetime import datetime
import re



app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search():
    try:
        data = request.json
        query = data.get('query')
        embedding_model = data.get('embedding_model', 'minilm')
        top_k = data.get('top_k', 5)
        ollama_model = data.get('ollama_model', 'mistral')

        if not query:
            return jsonify({'error': 'Query is required'}), 400

        # Get search results
        context_results, search_time, embedding_time = search_embeddings(
            collection, 
            query, 
            embedding_model=embedding_model, 
            top_k=top_k
        )

        # Generate RAG response
        response, response_time = generate_rag_response(
            query, 
            context_results, 
            ollama_model=ollama_model
        )

        return jsonify({
            'response': response,
            'context': context_results,
            'timing': {
                'embedding_time': embedding_time,
                'search_time': search_time,
                'response_time': response_time,
                'total_time': embedding_time + search_time + response_time
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

SAVE_PATH = "saved_recipes.json"

def load_saved_recipes():
    if not os.path.exists(SAVE_PATH):
        return []
    with open(SAVE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_all(recipes):
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(recipes, f, indent=2)

@app.route('/save', methods=['POST'])
def save():
    data = request.get_json()
    html = data.get('html', '')

    # 🧠 Try to auto-extract the first title as a fallback
    title_match = re.search(r'<strong>[^<]*?Dish that is recommended[^<]*?</strong>(.*?)<br>', html)
    fallback_title = "Untitled Recipe"
    extracted = html.split('<strong>')
    title_text = extracted[1].split('</strong>')[0].strip() if len(extracted) > 1 else fallback_title

    new_recipe = {
        "html": html,
        "title": title_text,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

    recipes = load_saved_recipes()
    recipes.append(new_recipe)
    save_all(recipes)

    return "Saved"

@app.route('/unsave', methods=['POST'])
def unsave():
    data = request.get_json()
    index = data.get('index')
    recipes = load_saved_recipes()
    if 0 <= index < len(recipes):
        del recipes[index]
        save_all(recipes)
        return "Unsave successful"
    return "Invalid index", 400

@app.route('/saved')
def show_saved():
    recipes = load_saved_recipes()  # returns a list of HTML recipe strings
    return render_template("saved_recipe.html", recipes=recipes)

@app.route('/rename', methods=['POST'])
def rename():
    data = request.get_json()
    index = data.get('index')
    new_title = data.get('new_title')

    recipes = load_saved_recipes()
    if 0 <= index < len(recipes):
        recipes[index]['title'] = new_title
        save_all(recipes)
        return "Renamed", 200

    return "Invalid index", 400


if __name__ == '__main__':
    app.run(debug=True, port=5001) 