from flask import Flask, render_template, request
from recommendation_engine import (
    get_recommendations,
    get_progress_chart,
    predict_performance
)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/recommend', methods=['POST'])
def recommend():
    student_id = int(request.form['student_id'])
    recommendations = get_recommendations(student_id)
    return render_template('recommend.html', student_id=student_id, recommendations=recommendations)

@app.route('/progress', methods=['POST'])
def progress():
    student_id = int(request.form['student_id'])
    chart_path = get_progress_chart(student_id)
    return render_template('progress.html', chart_path=chart_path, student_id=student_id)

@app.route('/predict', methods=['POST'])
def predict():
    student_id = int(request.form['student_id'])
    prediction = predict_performance(student_id)
    return render_template('prediction.html', student_id=student_id, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
