# from flask import Flask,request,jsonify
# from flask_cors import CORS
# import recommendation

# app = Flask(__name__)
# CORS(app) 
        
# @app.route('/movie', methods=['GET'])
# def recommend_movies():
#         res = recommendation.results(request.args.get('title'))
#         return jsonify(res)

# if __name__=='__main__':
#         app.run(port = 5000, debug = True)
        
# from flask import Flask,request,jsonify
# from flask_cors import CORS
# from recommender.modeling.spark_prediction import results

# app = Flask(__name__)
# CORS(app)    

# @app.route('/movie', methods=['GET'])

# def recommend_movies():
#     res = results(int(request.args.get('user')))
#     return jsonify(res)

# if __name__ == '__main__':
#     port = int(os.environ.get("PORT" or 5000))
#     app.run(host="0.0.0.0", port=port)

from flask import Flask,request,jsonify
from flask_cors import CORS
from recommender.modeling.prediction import results

app = Flask(__name__)
CORS(app) 
        
@app.route('/movie', methods=['GET'])
def recommend_movies():
        res = results(request.args.get('title'))
        return jsonify(res)

if __name__=='__main__':
        app.run(port = 5000, debug = True)