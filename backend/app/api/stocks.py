from ..db.influx_client import get_influxdb_client
from fastapi import APIRouter, HTTPException
from ..services.zerodha_service import ZerodhaService
import logging

# ZerodhaService instance to fetch live stock data
zerodha_service = ZerodhaService()

router = APIRouter()

logger = logging.getLogger(__name__)


@router.get("/stocks/{symbol}/data")
async def get_stock_data(symbol: str):
    try:
        client = get_influxdb_client()
        query_api = client.query_api()

        query = f'''
        from(bucket: "stock_data")
          |> range(start: -730d)
          |> filter(fn: (r) => r._measurement == "stock_data" and r.symbol == "{symbol}")
          |> keep(columns: ["_time", "_value", "_field", "symbol"])
        '''

        logger.info(f"Executing InfluxDB query for symbol: {symbol}")
        result = query_api.query(org="my_org", query=query)
        logger.info(f"Query result: {result}")

        stock_data = []
        for table in result:
            for record in table.records:
                stock_data.append({
                    "time": record["_time"],
                    "field": record["_field"],
                    "value": record["_value"]
                })

        logger.info(f"Processed {len(stock_data)} data points for {symbol}")
        return {"symbol": symbol, "data": stock_data}

    except Exception as e:
        logger.error(f"Error fetching stock data for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stocks/{symbol}/live")
async def get_live_stock_data(symbol: str):
    try:
        # Fetch live data for the stock symbol using Zerodha's API
        live_data = zerodha_service.get_quote(f"BSE:{symbol}")

        # Check if data is returned
        if not live_data or f"BSE:{symbol}" not in live_data:
            raise HTTPException(status_code=404, detail=f"No live data found for stock symbol {symbol}")

        # Extract and return the relevant data
        return live_data[f"BSE:{symbol}"]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching live data: {str(e)}")
