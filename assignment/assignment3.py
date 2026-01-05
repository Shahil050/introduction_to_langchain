from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI
from fastapi.responses import FileResponse

app = FastAPI()

# Base paths
BASE_DIR = Path.cwd().parent
RESOURCE_DIR = BASE_DIR / "resource"
DATA_PATH = RESOURCE_DIR / "prompts.csv"
CHART_PATH = RESOURCE_DIR / "prompt_analysis.png"


def analyze_prompts():
    # loading dataset
    df = pd.read_csv(DATA_PATH)

    # Optional safety check although i created the dataset correctly
    if not {"id", "prompt"}.issubset(df.columns):
        raise ValueError("CSV must contain 'id' and 'prompt' columns")

    # Counting tokens 
    df["token_count"] = df["prompt"].apply(lambda x: len(str(x).split()))

    # 3. Calculating  stats 
    stats = {
        "average": float(np.mean(df["token_count"])),
        "min": int(np.min(df["token_count"])),
        "max": int(np.max(df["token_count"])),
        "total_prompts": int(len(df)),
    }

    # Creating bar charts 
    plt.figure(figsize=(8, 4))
    plt.bar(df["id"].astype(str), df["token_count"], color="red")
    plt.xlabel("Prompt ID")
    plt.ylabel("Token Count")
    plt.title("Prompt Length Analysis")
    plt.tight_layout()

    # Ensuring the directory exist
    RESOURCE_DIR.mkdir(parents=True, exist_ok=True)

    # Saveing the created  chart
    plt.savefig(CHART_PATH)
    plt.close()

    return stats


@app.get("/analyze")
async def get_analysis():
    stats = analyze_prompts()
    return {
        "statistics": stats,
        "chart_url": "/chart",
    }


@app.get("/chart")
async def get_chart():
    return FileResponse(
        CHART_PATH,
        media_type="image/png",
        filename="prompt_analysis.png",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
