from joblib import load
from components.load import get_project_root, load_config




root = get_project_root()

images_path = f"{root}/components/images"

BR = load(f'{root}/models_ml/BR_model.joblib')
LR = load(f'{root}/models_ml/LinearRegression.joblib')
SR = load(f'{root}/models_ml/SR_model.joblib')
CatBoost = load(f'{root}/models_ml/CatBoostR_model.joblib')
GBR = load(f'{root}/models_ml/GBR_model.joblib')

readme = load_config("models_info.toml")
models = {
        'LinearRegression' : {
            'name_ru': "Линейная регрессия",
            'name_eng': "LinearRegression",
            'model': LR,
            'info': readme['linear']['description'],
            'metrics': {
                'MAE': 8.74,
                'MSE': 243.37,
                'RMSE': 15.6,
                'MAPE': 0.69,
                'R2': 0.66
            },

            'images_path': [f'{images_path}/lr.png']
        },
        'BaggingRegression':{
            'name_ru': "Ансамблевая модель бэггинга",
            'name_eng': "BaggingRegression",
            'model': BR,
            'info': readme['br']['description'],
            'metrics': {
                'MAE': 4.15,
                'MSE': 62.25,
                'RMSE': 7.89,
                'MAPE': 0.45,
                'R2': 0.88
            },
            'images_path': [f"{images_path}/bg.png"]
        },
        'StackingRegression':{
            'name_ru': 'Ансамблевая модель стэкинга',
            'name_eng': "StackingRegression",
            'model': SR,
            'info': readme['sr']['description'],
                'metrics': {
                'MAE': 4.78,
                'MSE': 81.47,
                'RMSE': 9.02,
                'MAPE': 0.48,
                'R2': 0.88
            },
            'images_path': [f"{images_path}/st.png"]
        },
        'GradientBoostingRegression': {
            'name_ru': 'Градиентный бустинг',
            'name_eng': "GradientBoostingRegression",
            'model': GBR,
            'info': readme['gbr']['description'],
                'metrics': {
                'MAE': 4.74,
                'MSE': 63.79,
                'RMSE': 7.98,
                'MAPE': 0.46,
                'R2': 0.90
            },
            'images_path': [f"{images_path}/GBM.png"]
        },
        'CatBoostRegression':{
            'name_ru': "Бустинг CatBoost",
            'name_eng': "CatBoostRegression",
            'model': CatBoost,
            'info': readme['catboost']['description'],
                'metrics': {
                'MAE': 5.95,
                'MSE': 107.08,
                'RMSE': 10.34,
                'MAPE': 0.54,
                'R2': 0.84
            }
        }
    }
def get_models():
    return models