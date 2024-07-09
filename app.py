from flask import Flask, request, render_template
from iris_model import dt_model, rf_model, lr_model

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        sepallength = float(request.form['sepallength'])
        sepalwidth = float(request.form['sepalwidth'])
        petallength = float(request.form['petallength'])
        petalwidth = float(request.form['petalwidth'])

        input_data = [[sepallength, sepalwidth, petallength, petalwidth]]

        dt_pred = dt_model.predict(input_data)[0]
        rf_pred = rf_model.predict(input_data)[0]
        lr_pred = lr_model.predict(input_data)[0]

        result = {
            'dt': dt_pred,
            'rf': rf_pred,
            'lr': lr_pred
        }

        return render_template('index.html', result=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)