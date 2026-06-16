import sys
from pathlib import Path

from flask import Flask,jsonify,request
from flask_cors import CORS


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))
print(BASE_DIR)

from src.recommend_bsd_on_genre import MovieRecommender

app = Flask(__name__)
CORS(app)

# Load model
movie_recommender = MovieRecommender()

@app.route("/",methods=["GET"])
def home():
    return jsonify({
        "success":True,
        "message":"Movie Reommendation API is running."
    })

@app.route("/api/movies",methods=["GET"])
def get_movies():
    limit = request.args.get("limit",default=25,type=int)
    page = request.args.get("page",default=1,type=int)
    query = request.args.get("q",default=None,type=str)

    limit = min(max(limit,1),100)
    page = max(page,1)

    movies,total = movie_recommender.get_movies(limit=limit,page=page,query=query)
    total_pages = max((total + limit - 1) // limit,1)

    return jsonify({
        "success":True,
        "count":len(movies),
        "total":total,
        "page":page,
        "limit":limit,
        "totalPages":total_pages,
        "hasPrev":page > 1,
        "hasNext":page < total_pages,
        "movies":movies
    })

@app.route("/api/movie/<int:movie_id_enc>",methods=["GET"])
def get_movie_details(movie_id_enc):
    movie = movie_recommender.get_movie_details(movie_id_enc)
    
    if movie is None:
        return jsonify({
            "success":False,
            "message":f"Movie with encoded Id {movie_id_enc} not found."
        }),404
    return jsonify({
        "success":True,
        "movie":movie
    })

@app.route("/api/recommend/movie/<int:movie_id_enc>",methods=["GET"])
def get_recommended_movies(movie_id_enc):
    top_n = request.args.get("top_n",default=10,type=int)

    result = movie_recommender.recommend_similar_movies(movie_id_enc,top_n)

    status_code = 200 if result["success"] else 404

    return jsonify(result),status_code


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
