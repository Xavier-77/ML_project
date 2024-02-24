from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Charger les modèles
classifier = load('classifier_model.joblib')
regressor = load('regressor_model.joblib')
#model = load('pipeline_model_combinee.joblib')

# Créer une classe de modèle Pydantic pour la requête JSON
class DataIn(BaseModel):
    Vn: float
    ZCR: float
    SF: float
    CGS: float
    SNR: float
    CS: float
    # Ajoutez d'autres caractéristiques au besoin

# Créer une classe de modèle Pydantic pour la réponse JSON
class DataOut(BaseModel):
    predicted_noise_category: str
    predicted_vocal_quality: float

# Instancier FastAPI
app = FastAPI()

# Définir la route POST pour la prédiction
@app.post("/predict/", response_model=DataOut)
def predict(data: DataIn):
    try:
        # Convertir les données d'entrée en tableau NumPy
        input_data = np.array([[data.Vn, data.ZCR, data.SF, data.CGS, data.SNR, data.CS]])

        # Utiliser les modèles pour prédire les catégories de bruit et la qualité vocale
        predicted_noise_category = str(classifier.predict(input_data)[0])
        predicted_vocal_quality = float(regressor.predict(input_data)[0])

        # Mapper les valeurs prédites à des noms de catégories de bruit
        if predicted_noise_category == 0:
            predicted_noise_category_str = "environnement"
        elif predicted_noise_category == 1:
            predicted_noise_category_str = "grésillement"
        else:
            predicted_noise_category_str = "souffle"
            
        if predicted_vocal_quality <= 1.25:
            predicted_vocal_quality_str = "médiocre"
        elif 1.25 < predicted_vocal_quality <= 3.5:
            predicted_vocal_quality_str = "moyenne"
        elif 3.5 < predicted_vocal_quality <= 4.5:
            predicted_vocal_quality_str = "bonne"
        else:
            predicted_vocal_quality_str = "excellente"

        # Créer et renvoyer la réponse au format JSON
        response_data = {
            "Le type de bruit est": predicted_noise_category_str,
            "La qualité vocale est": predicted_vocal_quality_str,
        }
        # Afficher les prédictions
        #print("Type de bruit prédit :", predicted_noise_category)
        #print("Qualité vocale prédite :", predicted_vocal_quality)

        # Créer et renvoyer la réponse au format JSON
        #response_data = {"Le type de bruit est": predicted_noise_category, "La qualité vocale est": predicted_vocal_quality}
        print("Données renvoyées :", response_data)
        return JSONResponse(content=response_data)
    except Exception as e:
        # Si une erreur se produit, renvoyer une réponse d'erreur avec un message approprié
        error_message = f"Une erreur s'est produite : {str(e)}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)

# Servir le fichier index.html
@app.get("/")
async def get_index():
    return FileResponse("index.html", media_type="text/html")
