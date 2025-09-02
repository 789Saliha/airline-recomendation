import pandas as pd

# Load dataset globally (so both app.py and module can use it)
df = pd.read_csv("airlines_reviews_clean.csv")

def hybrid_recommend(traveller_type, top_n=5):
    summary = df.groupby("Airline").agg(
        Avg_Rating=("Overall Rating", "mean"),
        Recommend_Rate=("Recommended", "mean"),
        Review_Count=("Airline", "count")
    ).reset_index()

    summary["Review_Count_Norm"] = summary["Review_Count"] / summary["Review_Count"].max()
    summary["Rule_Score"] = (
        0.5 * summary["Avg_Rating"] + 0.3 * summary["Recommend_Rate"] * 10 + 0.2 * summary["Review_Count_Norm"] * 10
    )

    summary["Hybrid_Score"] = summary["Rule_Score"]  # For simplicity, ML part skipped here
    summary = summary.sort_values(by="Hybrid_Score", ascending=False).head(top_n)

    return summary[["Airline", "Avg_Rating", "Rule_Score", "Hybrid_Score"]]
