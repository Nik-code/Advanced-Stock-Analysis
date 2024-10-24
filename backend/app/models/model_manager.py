import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta
import json
import shutil
from app.models.model_trainer import train_and_save_models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRAINED_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "trained_models")


async def get_or_train_model(stock_code):
    model_dir = os.path.join(TRAINED_MODELS_DIR, stock_code)
    metadata_path = os.path.join(model_dir, "metadata.json")

    if os.path.exists(model_dir) and os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        trained_date = datetime.fromisoformat(metadata["trained_date"])
        if datetime.now() - trained_date <= timedelta(days=7):
            logger.info(f"Using existing model for {stock_code}")
            return model_dir, metadata["metrics"]

        # If model is older than 7 days, delete the folder
        shutil.rmtree(model_dir)
        logger.info(f"Deleted old model for {stock_code}")

    logger.info(f"Training new model for {stock_code}")
    models = await train_and_save_models(stock_code)

    if not models:
        logger.error(f"Failed to train models for {stock_code}")
        return None, None

    os.makedirs(model_dir, exist_ok=True)

    best_model_name = min(models, key=lambda x: models[x].mse if models[x] is not None else float('inf'))
    best_model = models[best_model_name]

    metadata = {
        "trained_date": datetime.now().isoformat(),
        "best_model": best_model_name,
        "metrics": {
            model_name: {
                "MSE": getattr(model, 'mse', None),
                "MAE": getattr(model, 'mae', None)
            } for model_name, model in models.items() if model is not None
        }
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Model trained and saved for {stock_code}")
    return model_dir, metadata["metrics"]


async def load_model(stock_code):
    model_dir = os.path.join(TRAINED_MODELS_DIR, stock_code)
    model_path = os.path.join(model_dir, f"{stock_code}_model.pkl")
    metadata_path = os.path.join(model_dir, "metadata.json")

    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        logger.error(f"Model or metadata not found for {stock_code}")
        return None, None

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    best_model_name = metadata["best_model"]

    if best_model_name == "XGBoost":
        from app.models.XGBoost import XGBoostModel
        model = XGBoostModel.load(model_path)
    elif best_model_name == "ARIMA":
        from app.models.arima_model import ARIMAStockPredictor
        model = ARIMAStockPredictor.load(model_path)
    elif best_model_name == "RandomForest":
        from app.models.random_forest import RandomForestModel
        model = RandomForestModel.load(model_path)
    else:
        logger.error(f"Unknown model type: {best_model_name}")
        return None, None

    return model, best_model_name

if __name__ == "__main__":
    stock_code = "RELIANCE"  # You can change this to any stock code you want to test
    model_path, best_model_name = asyncio.run(get_or_train_model(stock_code))
    if model_path:
        logger.info(f"Model path: {model_path}")
        logger.info(f"Best model: {best_model_name}")

        # Load and test the model
        loaded_model, loaded_model_name = asyncio.run(load_model(stock_code))
        if loaded_model:
            logger.info(f"Successfully loaded {loaded_model_name} model for {stock_code}")
            # Add any testing code here
        else:
            logger.error("Failed to load model")
    else:
        logger.error("Failed to get or train model")
