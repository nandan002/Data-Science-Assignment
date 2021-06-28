from flask import Flask, request, jsonify
from extraction import *
import json

#### FLASK IS A MICRO WEB FRAMEWORK WRITTEN IN PYTHON
app = Flask(__name__)  # '__main__'


@app.route('/nc_keyword_extraction', methods=["POST"])
def noun_extraction():
    """This post method takes input - list of articles in json format
    and returns the top noun chunks from all the articles.
    """
    if request.method == 'GET':
        return "Please make POST request"

    elif request.method == 'POST':
        articles = []
        description = []
        params = request.get_json()
        list_articles = params['data']  # Storing the articles in list_articles
        for article in list_articles:
            for line in open(article, 'r'):  # Opening each article and loading all the json values in articles list
                articles.append(json.loads(line))

        for a in articles:  # Storing all the headline or text of the article in description list
            description.append(a['headline'])

        # Initialising the class NounExtract
        Noun = NounExtract()
        # Returns the list of articles after preprocessing
        preprocessed_data = list(map(Noun.preprocess, description))
        # Returns the bigrams and trigrams of the processed data
        vectorized_data = Noun.ngrams(preprocessed_data)
        # This returns the noun chunks of the bigrams and trigrams
        noun_chunks = Noun.noun_extract(vectorized_data)
        # This returns the Top noun words in descending order
        top_n_words = Noun.top_n_words(noun_chunks)
        # This returns final noun words after post-processing
        final_words = Noun.final_processing(top_n_words)[:10]

    # Returning the final output in json format
    return jsonify({"noun_chunks": {"nc": final_words}})


if __name__ == '__main__':
    app.run(debug=True)
