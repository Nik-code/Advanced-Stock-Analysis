import sys
import os
import asyncio
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.model_trainer import train_and_save_models, compare_models
from app.models.model_manager import get_or_train_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_models(stock_code):
    logger.info(f"Getting or training model for {stock_code}...")
    model_path = await get_or_train_model(stock_code)

    if model_path is None:
        logger.error(f"Failed to get or train model for {stock_code}")
        return

    logger.info(f"Model path: {model_path}")

    logger.info(f"Comparing models for {stock_code}...")
    results = await compare_models(stock_code)

    if results is None:
        logger.error(f"Failed to compare models for {stock_code}")
    else:
        logger.info(f"Model comparison results for {stock_code}:")
        for model_name, metrics in results.items():
            logger.info(f"{model_name}: MSE = {metrics['MSE']:.4f}, MAE = {metrics['MAE']:.4f}")


if __name__ == "__main__":
    stock_code = "SUNPHARMA"  # You can change this to any stock code you want to test
    asyncio.run(test_models(stock_code))
