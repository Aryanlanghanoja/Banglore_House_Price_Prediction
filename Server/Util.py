import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None


def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())

    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = bhk
    x[1] = sqft
    x[2] = bath

    if loc_index >= 0:
        x[loc_index] = 1

    return np.round(__model.predict([x])[0], 2)


def Load_Saved_Artifiacts():
    print("Loading Saved Artifacts... Start")
    global __data_columns
    global __locations
    global __model

    with open("Artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)["data_columns"]
        __locations = __data_columns[3:]
        f.close()

    if __model is None:
        with open("Artifacts/banglore_home_prices_model.pickle", "rb") as f:
            __model = pickle.load(f)
            f.close()

    print("Loading Artifacts are Done")


def get_data_columns():
    return __data_columns


def get_location_names():
    return __locations


if __name__ == "__main__":
    Load_Saved_Artifiacts()
    print(Load_Saved_Artifiacts())
    print(get_estimated_price("1st Phase JP Nagar", 1000, 3, 3))
    print(get_estimated_price("1st Phase JP Nagar", 1000, 2, 2))
    print(get_estimated_price("Kalhalli", 1000, 2, 2))  # other location
    print(get_estimated_price("Ejipura", 1000, 2, 2))  # other location
