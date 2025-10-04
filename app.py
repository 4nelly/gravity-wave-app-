from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Загрузка модели и скейлера
model = joblib.load('xgboost_model.joblib')
scaler = joblib.load('scaler_XGBoost.joblib')

def get_comment(final_mass, sum_mass):
    """Генерирует интересный комментарий на основе предсказания."""
    ratio = final_mass / sum_mass
    if sum_mass < 5:
        return "Это слияние нейтронных звёзд! Такие события создают яркие гамма-всплески, видимые через телескопы!"
    elif sum_mass < 20:
        return "Вероятно, слияние нейтронной звезды и чёрной дыры — редкое космическое явление!"
    else:
        return "Массивные чёрные дыры слились! Это вызывает мощные гравитационные волны, которые регистрирует LIGO!"

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    comment = None
    error = None
    
    if request.method == 'POST':
        try:
            # Получение входных данных
            mass_1 = float(request.form['mass_1'])
            mass_2 = float(request.form['mass_2'])
            
            if mass_1 <= 0 or mass_2 <= 0:
                error = "Массы должны быть положительными числами!"
            else:
                # Вычисление Sum_mass
                sum_mass = mass_1 + mass_2
                
                # Подготовка данных для модели
                sum_mass_scaled = scaler.transform(np.array([[sum_mass]]))
                
                # Предсказание
                final_mass = model.predict(sum_mass_scaled)[0]
                
                # Постобработка: ограничение диапазона [0.9*Sum_mass, 0.99*Sum_mass]
                final_mass = np.clip(final_mass, 0.9 * sum_mass, 0.99 * sum_mass)
                
                # Округление результата
                prediction = round(final_mass, 2)
                
                # Генерация комментария
                comment = get_comment(final_mass, sum_mass)
                
        except ValueError:
            error = "Пожалуйста, введите корректные числа для масс!"
    
    return render_template('index.html', prediction=prediction, comment=comment, error=error)

if __name__ == '__main__':
    app.run(debug=True)