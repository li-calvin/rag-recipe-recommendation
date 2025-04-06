from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from Database.query import search_embeddings, generate_rag_response, chroma_client, collection

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

if __name__ == '__main__':
    app.run(debug=True, port=5001) 