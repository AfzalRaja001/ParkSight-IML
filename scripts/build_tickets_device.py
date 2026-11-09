"""Build research_data/tickets_device.parquet (gitignored): id, police_station, date, hour, device_id.
Same UTC -> IST conversion as parksight/etl/build_artifacts.py."""
import pandas as pd

from parksight import config as C

df = pd.read_csv(C.RAW_CSV, usecols=["id", "police_station", "created_datetime", "device_id"], low_memory=False)
ist = pd.to_datetime(df["created_datetime"], utc=True, errors="coerce").dt.tz_convert(C.TZ_IST)
out = pd.DataFrame({"id": df["id"], "police_station": df["police_station"],
                    "date": ist.dt.tz_localize(None).dt.normalize(), "hour": ist.dt.hour,
                    "device_id": df["device_id"]})[ist.notna()]
C.ROOT.joinpath("research_data").mkdir(exist_ok=True)
out.to_parquet(C.ROOT / "research_data" / "tickets_device.parquet", index=False)
print(len(out), out["device_id"].nunique(), "devices;", out["date"].min(), "->", out["date"].max())
