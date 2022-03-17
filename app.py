from flask import Flask, render_template, request
import joblib 

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods = ["post"])
def predict():
    area = request.form.get('area')
    rooms = request.form.get('rooms')
    bathroom = request.form.get('bathroom')
    floors = request.form.get('floors')
    driveway = request.form.get('driveway')
    game_room = request.form.get('game_room')
    cellar = request.form.get('cellar')
    gas = request.form.get('gas')
    air = request.form.get('air')
    garage = request.form.get('garage')
    situation = request.form.get('situation')
    print(area,rooms,bathroom,floors,driveway,game_room,cellar,gas,air,garage,situation)

    model = joblib.load('houseprice_63.pkl')
    
    data = model.predict([[area,rooms,bathroom,floors,driveway,game_room,cellar,gas,air,garage,situation]])
    print(data[0,0])

    return render_template('predict.html', output = data[0,0])

app.run(debug=True)