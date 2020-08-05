from flask import Flask, jsonify
from api_v1 import api


@api.route('/facheck', methods=['POST'])
def facheck():
    return jsonify({'text': translate(request.form['text'],
                                      request.form['source_language'],
                                      request.form['dest_language'])})
