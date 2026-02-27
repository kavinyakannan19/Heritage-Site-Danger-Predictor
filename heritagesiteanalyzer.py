# =====================================================
# FINAL ECO RISK WEB SYSTEM (ANIMATED UI VERSION)
# =====================================================

from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

app = Flask(__name__)

# =====================================================
# CLEAN LATITUDE / LONGITUDE
# =====================================================
def clean_coordinate(value):
    value = str(value).strip()
    if value.endswith("N"):
        return float(value[:-1])
    elif value.endswith("S"):
        return -float(value[:-1])
    elif value.endswith("E"):
        return float(value[:-1])
    elif value.endswith("W"):
        return -float(value[:-1])
    else:
        return float(value)

# =====================================================
# UNESCO MODEL
# =====================================================

unesco = pd.read_csv("whc-sites-2025.csv")

unesco = unesco[[
    "name_en","latitude","longitude",
    "category","states_name_en","danger"
]].dropna()

unesco.rename(columns={"states_name_en":"country"}, inplace=True)

unesco["latitude"] = unesco["latitude"].apply(clean_coordinate)
unesco["longitude"] = unesco["longitude"].apply(clean_coordinate)
unesco["danger"] = unesco["danger"].replace({"Yes":1,"No":0})

le_cat = LabelEncoder()
le_country = LabelEncoder()

unesco["category"] = le_cat.fit_transform(unesco["category"])
unesco["country"] = le_country.fit_transform(unesco["country"])

X_unesco = unesco[["latitude","longitude","category","country"]]
y_unesco = unesco["danger"]

model_unesco = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)
model_unesco.fit(X_unesco,y_unesco)

# =====================================================
# TOURISM MODEL
# =====================================================

tourism = pd.read_csv("tourism_dataset.csv")

tourism = tourism[[
    "Location","Country","Category",
    "Visitors","Revenue","Rating",
    "Accommodation_Available"
]].dropna()

tourism["Accommodation_Available"] = tourism["Accommodation_Available"].map({"Yes":1,"No":0})

le_tour_country = LabelEncoder()
le_tour_cat = LabelEncoder()

tourism["Country"] = le_tour_country.fit_transform(tourism["Country"])
tourism["Category"] = le_tour_cat.fit_transform(tourism["Category"])

X_tour = tourism[[
    "Country","Category","Visitors",
    "Revenue","Accommodation_Available"
]]

y_tour = tourism["Rating"]

model_tour = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
model_tour.fit(X_tour,y_tour)

# =====================================================
# TEMPERATURE MODEL
# =====================================================

temp = pd.read_csv("GlobalLandTemperaturesByMajorCity.csv")

temp = temp[[
    "Country","Latitude","Longitude",
    "AverageTemperature",
    "AverageTemperatureUncertainty"
]].dropna()

temp["Latitude"] = temp["Latitude"].apply(clean_coordinate)
temp["Longitude"] = temp["Longitude"].apply(clean_coordinate)

le_temp_country = LabelEncoder()
temp["Country"] = le_temp_country.fit_transform(temp["Country"])

X_temp = temp[[
    "Country","Latitude","Longitude",
    "AverageTemperatureUncertainty"
]]

y_temp = temp["AverageTemperature"]

model_temp = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
model_temp.fit(X_temp,y_temp)

mean_temp = y_temp.mean()

# =====================================================
# ANIMATED HTML UI
# =====================================================

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>ECO Risk Analyzer</title>

<style>

*{
margin:0;
padding:0;
box-sizing:border-box;
}

body{
height:100vh;
display:flex;
justify-content:center;
align-items:center;
font-family: 'Segoe UI', sans-serif;
background: linear-gradient(-45deg,#0f2027,#203a43,#2c5364,#00c9ff);
background-size:400% 400%;
animation: gradientBG 12s ease infinite;
overflow:hidden;
}

@keyframes gradientBG{
0%{background-position:0% 50%;}
50%{background-position:100% 50%;}
100%{background-position:0% 50%;}
}

@keyframes floatBox{
0%{transform:translateY(0px);}
50%{transform:translateY(-10px);}
100%{transform:translateY(0px);}
}

.container{
background:rgba(255,255,255,0.12);
backdrop-filter:blur(20px);
-webkit-backdrop-filter:blur(20px);
padding:40px;
border-radius:25px;
width:520px;
box-shadow:0 20px 50px rgba(0,0,0,0.5);
text-align:center;
color:white;
animation:floatBox 5s ease-in-out infinite;
transition:0.4s;
}

h1{
margin-bottom:20px;
font-size:28px;
}

input,select{
width:100%;
padding:12px;
margin:10px 0;
border-radius:10px;
border:none;
outline:none;
font-size:14px;
}

input:focus,select:focus{
box-shadow:0 0 10px #00f2ff;
}

button{
width:100%;
padding:14px;
margin-top:15px;
border:none;
border-radius:10px;
cursor:pointer;
font-size:16px;
font-weight:bold;
color:white;
background:linear-gradient(45deg,#00c9ff,#92fe9d);
transition:0.4s;
}

button:hover{
transform:scale(1.05);
box-shadow:0 0 15px #00f2ff;
}

.result-box{
margin-top:25px;
padding:20px;
border-radius:15px;
background:rgba(0,0,0,0.3);
animation:fadeIn 1s ease-in-out;
}

@keyframes fadeIn{
from{opacity:0; transform:translateY(10px);}
to{opacity:1; transform:translateY(0);}
}

.percent{
font-size:42px;
font-weight:bold;
background: linear-gradient(45deg,#00f2ff,#00ff88);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
animation:pulse 2s infinite;
}

@keyframes pulse{
0%{transform:scale(1);}
50%{transform:scale(1.1);}
100%{transform:scale(1);}
}

</style>
</head>

<body>

<div class="container">

<h1>🌍 ECO Risk Analyzer</h1>

<form method="POST">
<input name="place" placeholder="Place Name" required>
<input name="country" placeholder="Country" required>
<input name="category" placeholder="Category" required>
<input type="number" step="any" name="latitude" placeholder="Latitude" required>
<input type="number" step="any" name="longitude" placeholder="Longitude" required>
<input type="number" step="any" name="visitors" placeholder="Visitors" required>
<input type="number" step="any" name="revenue" placeholder="Revenue" required>

<select name="accommodation">
<option value="Yes">Accommodation Available</option>
<option value="No">No Accommodation</option>
</select>

<input type="number" step="any" name="uncertainty" placeholder="Temperature Uncertainty" required>

<button type="submit">Analyze Safety</button>
</form>

{% if result %}
<div class="result-box">
<h2>{{ result.place }}</h2>
<div class="percent">{{ result.overall }} %</div>
</div>
{% endif %}

</div>

</body>
</html>
"""

# =====================================================
# ROUTE
# =====================================================

@app.route("/", methods=["GET","POST"])
def index():

    result = None

    if request.method == "POST":

        place = request.form["place"]
        country = request.form["country"]
        category = request.form["category"]
        latitude = float(request.form["latitude"])
        longitude = float(request.form["longitude"])
        visitors = float(request.form["visitors"])
        revenue = float(request.form["revenue"])
        uncertainty = float(request.form["uncertainty"])
        accommodation = request.form["accommodation"]

        acc_val = 1 if accommodation=="Yes" else 0

        cu = le_country.transform([country])[0] if country in le_country.classes_ else 0
        cat_u = le_cat.transform([category])[0] if category in le_cat.classes_ else 0
        ct = le_tour_country.transform([country])[0] if country in le_tour_country.classes_ else 0
        cat_t = le_tour_cat.transform([category])[0] if category in le_tour_cat.classes_ else 0
        ctemp = le_temp_country.transform([country])[0] if country in le_temp_country.classes_ else 0

        unesco_df = pd.DataFrame([[latitude,longitude,cat_u,cu]],
            columns=["latitude","longitude","category","country"])
        danger = model_unesco.predict(unesco_df)[0]
        unesco_score = 0 if danger==1 else 30

        tour_df = pd.DataFrame([[ct,cat_t,visitors,revenue,acc_val]],
            columns=["Country","Category","Visitors","Revenue","Accommodation_Available"])
        predicted_rating = model_tour.predict(tour_df)[0]
        predicted_rating = max(0,min(5,predicted_rating))
        tourism_score = (predicted_rating/5)*35

        temp_df = pd.DataFrame([[ctemp,latitude,longitude,uncertainty]],
            columns=["Country","Latitude","Longitude","AverageTemperatureUncertainty"])
        temp_pred = model_temp.predict(temp_df)[0]
        diff = abs(temp_pred-mean_temp)

        if diff <=2:
            climate_score = 35
        elif diff<=5:
            climate_score = 25
        elif diff<=8:
            climate_score = 15
        else:
            climate_score = 5

        overall = round(unesco_score + tourism_score + climate_score,2)

        result = {
            "place": place,
            "overall": overall
        }

    return render_template_string(HTML,result=result)

if __name__ == "__main__":
    app.run(debug=True)