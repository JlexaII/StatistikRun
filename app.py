from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

def perform_regression(file):
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            return {"error": "Поддерживаются только CSV и Excel файлы."}

        if 'Y' not in df.columns or not any(col.startswith('X') for col in df.columns):
            return {"error": "Файл должен содержать столбец 'Y' и хотя бы один столбец 'X'."}

        Y = df["Y"].values.reshape(-1, 1)
        X_cols = [col for col in df.columns if col.startswith('X')]
        X = df[X_cols].values
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        XTX = X.T @ X
        XTY = X.T @ Y
        B = np.linalg.inv(XTX) @ XTY
        Y_hat = X @ B

        SSR = np.sum((Y_hat - Y.mean())**2)
        SSE = np.sum((Y - Y_hat)**2)
        SST = np.sum((Y - Y.mean())**2)
        R2 = SSR / SST
        R = np.sqrt(R2)

        eq = f"Y = {B[0][0]:.4f}"
        for i in range(1, len(B)):
            eq += f" + ({B[i][0]:.4f})*X{i}"

        chart_data = {
            "labels": list(range(len(Y))),
            "actual": Y.flatten().tolist(),
            "predicted": Y_hat.flatten().tolist()
        }

        return {
            "coefficients": B.flatten().tolist(),
            "R2": R2,
            "R": R,
            "SSE": SSE,
            "SSR": SSR,
            "SST": SST,
            "equation": eq,
            "chart_data": chart_data
        }
    except Exception as e:
        return {"error": f"Ошибка: {str(e)}"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "Файл не загружен."}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Файл не выбран."}), 400

    result = perform_regression(file)
    return jsonify(result)
