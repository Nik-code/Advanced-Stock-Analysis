import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.model_trainer import train_and_save_models, compare_models


async def test_models(stock_code):
    print(f"Training and saving models for {stock_code}...")
    models = await train_and_save_models(stock_code)

    if models is None:
        print(f"Failed to train models for {stock_code}")
        return

    print(f"Comparing models for {stock_code}...")
    results = await compare_models(stock_code)

    if results is None:
        print(f"Failed to compare models for {stock_code}")
    else:
        print(f"Model comparison results for {stock_code}:")
        for model_name, metrics in results.items():
            print(f"{model_name}: MSE = {metrics['MSE']:.4f}, MAE = {metrics['MAE']:.4f}")


if __name__ == "__main__":
    stock_code = "RELIANCE"  # You can change this to any stock code you want to test
    asyncio.run(test_models(stock_code))
