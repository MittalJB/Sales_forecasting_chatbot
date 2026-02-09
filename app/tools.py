import pandas as pd

df1 = pd.read_csv("data/features.csv")
df2 = pd.read_csv("data/stores.csv")
df3 = pd.read_csv("data/test.csv")
df4 = pd.read_csv("data/train.csv")

# Merge datasets
data = df4.merge(df2, on="Store", how="left")
data = data.merge(df1, on=["Store", "Date"], how="left")

# ---------------- FIX IsHoliday ---------------- #
# Prefer holiday flag if it exists in either dataset
if "IsHoliday_x" in data.columns or "IsHoliday_y" in data.columns:
    data["IsHoliday"] = (
        data.get("IsHoliday_x", False).fillna(False) |
        data.get("IsHoliday_y", False).fillna(False)
    )

# Drop duplicates
data.drop(
    columns=[c for c in ["IsHoliday_x", "IsHoliday_y"] if c in data.columns],
    inplace=True
)

# Convert TRUE/FALSE strings to boolean
if data["IsHoliday"].dtype == object:
    data["IsHoliday"] = data["IsHoliday"].map({"TRUE": True, "FALSE": False})

# Fill NA for MarkDowns with 0
for col in ["MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5"]:
    if col in data.columns:
        data[col] = data[col].fillna(0)

print(data.head())

# ---------------- TOOLS ---------------- #

def sales_summary(store=None, dept=None, category=None):
    df = data
    if store:
        df = df[df["Store"] == store]
    if dept:
        df = df[df["Dept"] == dept]

    if df.empty:
        return "No sales data available for the selected filters."

    total_sales = df["Weekly_Sales"].sum()
    avg_sales = df["Weekly_Sales"].mean()

    return (
        f"Total sales are {total_sales:,.2f} with an average weekly sales of "
        f"{avg_sales:,.2f}."
    )

def recent_drop(store=None, dept=None):
    df = data
    if store:
        df = df[df["Store"] == store]
    if dept:
        df = df[df["Dept"] == dept]

    if len(df) < 2:
        return "Not enough data to detect recent sales trends."

    df_sorted = df.sort_values("Date")
    last_week = df_sorted["Weekly_Sales"].iloc[-1]
    prev_week = df_sorted["Weekly_Sales"].iloc[-2]

    pct_change = ((last_week - prev_week) / prev_week) * 100

    if pct_change < -10:
        return f"Sales dropped sharply by {pct_change:.1f}% last week."
    return "No significant sales drop detected last week."

def markdown_impact(store=None, dept=None):
    df = data.copy()
    if store:
        df = df[df["Store"] == store]
    if dept:
        df = df[df["Dept"] == dept]

    if df.empty:
        return "No data available to analyze markdown impact."

    df["total_markdown"] = df[
        ["MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5"]
    ].sum(axis=1)

    correlation = df["Weekly_Sales"].corr(df["total_markdown"])

    return (
        f"The correlation between markdowns and sales is {correlation:.2f}. "
        "Higher positive values indicate markdowns likely boosted sales."
    )

def holiday_impact(store=None, dept=None):
    df = data
    if store:
        df = df[df["Store"] == store]
    if dept:
        df = df[df["Dept"] == dept]

    if df.empty:
        return "No data available to analyze holiday impact."

    holiday_avg = df[df["IsHoliday"] == True]["Weekly_Sales"].mean()
    non_holiday_avg = df[df["IsHoliday"] == False]["Weekly_Sales"].mean()

    if pd.isna(holiday_avg) or pd.isna(non_holiday_avg):
        return "Insufficient data to compare holiday and non-holiday sales."

    diff_pct = ((holiday_avg - non_holiday_avg) / non_holiday_avg) * 100

    if diff_pct > 0:
        return (
            f"Holidays increase sales by approximately {diff_pct:.1f}%. "
            f"Average holiday sales are {holiday_avg:,.2f} compared to "
            f"{non_holiday_avg:,.2f} during non-holiday weeks."
        )
    else:
        return (
            f"Holidays reduce sales by approximately {abs(diff_pct):.1f}%. "
            f"Merchants may need additional promotions during holiday weeks."
        )
