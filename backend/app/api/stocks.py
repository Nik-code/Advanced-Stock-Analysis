from fastapi import APIRouter, HTTPException
from ..db.influx_client import get_influxdb_client


router = APIRouter()

@router.get("/stocks/{symbol}/data")
async def get_stock_data(symbol: str):
    try:
        client = get_influxdb_client()
        query_api = client.query_api()

        query = f'''
        from(bucket: "stock_data")
          |> range(start: -30d)
          |> filter(fn: (r) => r._measurement == "stock_data" and r.symbol == "{symbol}")
          |> keep(columns: ["_time", "_value", "_field", "symbol"])
        '''

        result = query_api.query(org="my_org", query=query)

        stock_data = []
        for table in result:
            for record in table.records:
                stock_data.append({
                    "time": record["_time"],
                    "field": record["_field"],
                    "value": record["_value"]
                })

        return {"symbol": symbol, "data": stock_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
