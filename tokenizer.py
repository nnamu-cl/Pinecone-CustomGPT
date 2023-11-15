from flask import Flask, request, jsonify
from openai import OpenAI

import pinecone

app = Flask(__name__)

# Set your OpenAI API key here
openai_client = OpenAI(api_key = 'YOUR API KEY GOES HERE')

pinecone_api_key = 'YOUR API KEY GOES HERE'
index_name = 'YOUR index GOES HERE'
environment = 'YOUR env GOES HERE'

pinecone.init(api_key=pinecone_api_key, environment=environment)
index = pinecone.Index(index_name)


@app.route('/tokenize', methods=['GET'])
def tokenize():
    # Retrieve text from query parameter
    text = request.args.get('text')

    # Using OpenAI's embedding feature
    response = openai_client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"  # Choose the appropriate embedding model
    )

    # Extracting the embedding vector
    embedding_vector = response.data[0].embedding

    query_results = index.query(embedding_vector, top_k=5, include_metadata=True)

    results = query_results.to_dict()

    texts = [match['metadata']['text'] for match in results['matches']]

    return jsonify(texts)


if __name__ == '__main__':
    app.run(debug=True, port=80, host='0.0.0.0')
