from flask import Flask, request, jsonify
import Util

app = Flask(__name__)


@app.route("/")
def home():
    return "Welcome to the Home Price Prediction API"


@app.route("/Get_Location_Names", methods=["GET"])
def Get_Location_Names():
    response = jsonify({"Locations": Util.get_location_names()})

    response.headers.add("Access-Control-Allow-Origin", "*")

    return response


@app.route("/predict_home_price", methods=["GET", "POST"])
def predict_home_price():
    total_sqft = float(request.form["total_sqft"])
    location = request.form["location"]
    bhk = int(request.form["bhk"])
    bath = int(request.form["bath"])

    response = jsonify(
        {
            "estimated_price": Util.get_estimated_price(
                location, total_sqft, bhk, bath
            ).item()
        }
    )

    response.headers.add("Access-Control-Allow-Origin", "*")

    return response


if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction....")
    Util.Load_Saved_Artifiacts()
    app.run()
