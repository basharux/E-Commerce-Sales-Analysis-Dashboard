import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Online Retail Analytics Dashboard",
    page_icon="🛒",
    layout="wide"
)

# =====================================================
# CONSTANTS
# =====================================================
CLEAN_FILE = "online_retail_clean.csv.gz"
SUMMARY_FILE = "cleaning_summary.json"

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"]

ACCENT = "#00B4D8"
GREEN = "#2ECC71"
RED = "#E74C3C"

SEGMENT_ORDER = ["Champions", "Loyal", "New Customers",
                 "Need Attention", "At Risk", "Lost"]
SEGMENT_COLORS = {
    "Champions": "#2ECC71",
    "Loyal": "#3498DB",
    "New Customers": "#9B59B6",
    "Need Attention": "#F1C40F",
    "At Risk": "#E67E22",
    "Lost": "#E74C3C",
}


# =====================================================
# STEP 1: FIND + READ THE RAW FILE
# =====================================================
def find_raw_file():
    """Find the Online Retail Excel/CSV file in the current folder."""
    for f in os.listdir("."):
        name = f.lower()
        if (name.endswith((".xlsx", ".csv"))
                and "online" in name
                and "retail" in name
                and "clean" not in name):
            return f
    return None


def read_raw(path):
    if path.lower().endswith(".xlsx"):
        return pd.read_excel(
            path, dtype={"InvoiceNo": str, "StockCode": str}
        )
    return pd.read_csv(
        path, encoding="latin-1", parse_dates=["InvoiceDate"],
        dtype={"InvoiceNo": str, "StockCode": str}
    )


# =====================================================
# STEP 2: DATA CLEANING
# =====================================================
def clean_data(raw):
    stats = {"raw_rows": int(len(raw))}
    df = raw.copy()

    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()

    # 1. Remove duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    stats["duplicates_removed"] = int(before - len(df))

    # 2. Missing CustomerID (kept, but excluded from customer metrics)
    stats["missing_customer_rows"] = int(df["CustomerID"].isna().sum())

    # 3. Remove non-product rows (POST, DOT, BANK CHARGES, gift vouchers...)
    before = len(df)
    df = df[df["StockCode"].str.match(r"^\d{5}")]
    stats["non_product_removed"] = int(before - len(df))

    # 4. Cancellations (invoice starts with C) vs valid sales
    is_cancel = df["InvoiceNo"].str.upper().str.startswith("C")
    sale_ok = (~is_cancel) & (df["Quantity"] > 0) & (df["UnitPrice"] > 0)
    cancel_ok = is_cancel & (df["Quantity"] < 0) & (df["UnitPrice"] > 0)

    stats["invalid_removed"] = int((~(sale_ok | cancel_ok)).sum())
    df = df[sale_ok | cancel_ok].copy()
    df["Type"] = (
        df["InvoiceNo"].str.upper().str.startswith("C")
        .map({True: "Cancellation", False: "Sale"})
    )

    # 5. One clean description per product (most common one)
    names = (
        df.dropna(subset=["Description"])
        .assign(Description=lambda d: d["Description"].astype(str).str.strip().str.upper())
        .groupby(["StockCode", "Description"]).size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .drop_duplicates("StockCode")
        .set_index("StockCode")["Description"]
    )
    df["Description"] = df["StockCode"].map(names)
    before = len(df)
    df = df.dropna(subset=["Description"])
    stats["no_description_removed"] = int(before - len(df))

    # 6. Types and revenue
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce").astype("Int64")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Revenue"] = (df["Quantity"] * df["UnitPrice"]).round(2)

    df = df.sort_values("InvoiceDate").reset_index(drop=True)
    df = df[["InvoiceNo", "StockCode", "Description", "Quantity",
             "InvoiceDate", "UnitPrice", "CustomerID", "Country",
             "Type", "Revenue"]]

    stats["final_rows"] = int(len(df))
    stats["sales_rows"] = int((df["Type"] == "Sale").sum())
    stats["cancellation_rows"] = int((df["Type"] == "Cancellation").sum())
    stats["start_date"] = str(df["InvoiceDate"].min().date())
    stats["end_date"] = str(df["InvoiceDate"].max().date())

    return df, stats


# =====================================================
# HELPER FUNCTIONS
# =====================================================
def fmt_money(v):
    return f"£{v:,.0f}"


def quantile_score(series, reverse=False):
    """Score 1-5 using quintiles (ties get the same score)."""
    edges = series.quantile([0.2, 0.4, 0.6, 0.8]).values
    score = np.searchsorted(edges, series.values, side="left") + 1
    return 6 - score if reverse else score


def month_stats(sdf, month):
    d = sdf[sdf["Month"] == month]
    revenue = d["Revenue"].sum()
    orders = d["InvoiceNo"].nunique()
    return {
        "revenue": revenue,
        "orders": orders,
        "customers": d["CustomerID"].nunique(),
        "aov": revenue / orders if orders else 0,
    }


def delta_str(cur, prev):
    if prev is None or prev == 0:
        return None
    return f"{(cur - prev) / prev * 100:+.1f}% vs last month"


def style_fig(fig, height=None):
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    if height:
        fig.update_layout(height=height)
    return fig


# ===== UI STARTS HERE =====

# =====================================================
# STEP 3: BUILD CLEAN FILE ON FIRST RUN
# =====================================================
if not os.path.exists(CLEAN_FILE):
    raw_path = find_raw_file()

    if raw_path is None:
        st.error(
            "Online Retail file not found. Put the Excel file "
            "(for example 'Online Retail.xlsx') in the same folder as this "
            "script and run again."
        )
        st.stop()

    with st.spinner(
        "First run only: reading and cleaning the data. "
        "This can take 1-3 minutes..."
    ):
        raw_df = read_raw(raw_path)
        clean_df, clean_stats = clean_data(raw_df)
        clean_df.to_csv(CLEAN_FILE, index=False, compression="gzip")
        with open(SUMMARY_FILE, "w") as f:
            json.dump(clean_stats, f)


@st.cache_data(show_spinner="Loading data...")
def load_clean():
    d = pd.read_csv(
        CLEAN_FILE,
        parse_dates=["InvoiceDate"],
        dtype={"InvoiceNo": str, "StockCode": str, "CustomerID": "Int64"}
    )
    d["Month"] = d["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    d["Weekday"] = d["InvoiceDate"].dt.day_name()
    d["Hour"] = d["InvoiceDate"].dt.hour
    return d


df = load_clean()
data_end = df["InvoiceDate"].max()

# Theme: background, sidebar, cards, tabs
st.markdown("""
<style>
/* page background */
.stApp {
    background: linear-gradient(160deg, #0A1128 0%, #0F1F3D 45%, #0B2545 100%);
    background-attachment: fixed;
}
[data-testid="stHeader"] { background: rgba(0, 0, 0, 0); }

/* sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2E 0%, #0A1426 100%);
    border-right: 1px solid rgba(0, 180, 216, 0.25);
}

/* KPI cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(0, 180, 216, 0.14) 0%, rgba(30, 50, 90, 0.55) 100%);
    border: 1px solid rgba(0, 180, 216, 0.28);
    border-left: 4px solid #00B4D8;
    padding: 14px 16px;
    border-radius: 12px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
}
[data-testid="stMetricValue"] { font-size: 1.65rem; }
[data-testid="stMetricValue"] > div {
    overflow: visible;
    text-overflow: clip;
    white-space: nowrap;
}

/* chart panels */
[data-testid="stPlotlyChart"] {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 12px;
    padding: 6px;
}

/* tabs */
button[data-baseweb="tab"] { font-weight: 600; }
button[data-baseweb="tab"][aria-selected="true"] { color: #00B4D8; }
[data-baseweb="tab-highlight"] { background-color: #00B4D8; }

/* filter tags */
span[data-baseweb="tag"] { background-color: #0077B6; }

/* headings, dividers, expanders */
h1 { color: #E8F4FA; }
h2, h3 { color: #CDEBF5; }
hr { border-color: rgba(0, 180, 216, 0.2); }
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(0, 180, 216, 0.2);
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# TITLE + SIDEBAR FILTERS
# =====================================================
st.title("🛒 Online Retail Analytics Dashboard")
st.caption(
    "Real transactions of a UK-based online retailer "
    "(public UCI / Kaggle 'Online Retail' dataset). Currency: £"
)

st.sidebar.title("🔍 Filters")

min_d = df["InvoiceDate"].min().date()
max_d = df["InvoiceDate"].max().date()

date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_d, max_d),
    min_value=min_d,
    max_value=max_d
)

countries = sorted(df["Country"].unique())
sel_countries = st.sidebar.multiselect(
    "Country (leave empty for all)",
    countries,
    placeholder="All countries"
)

if len(date_range) != 2:
    st.info("Please select both a start date and an end date.")
    st.stop()

start_d, end_d = date_range

if sel_countries:
    country_mask = df["Country"].isin(sel_countries)
else:
    country_mask = pd.Series(True, index=df.index)

date_mask = (
    (df["InvoiceDate"] >= pd.Timestamp(start_d))
    & (df["InvoiceDate"] < pd.Timestamp(end_d) + pd.Timedelta(days=1))
)

base = df[country_mask]                  # country filter only
base_sales = base[base["Type"] == "Sale"]

fdf = df[country_mask & date_mask]       # country + date filter
sales = fdf[fdf["Type"] == "Sale"]
canc = fdf[fdf["Type"] == "Cancellation"]

if sales.empty:
    st.warning("No sales for the selected filters.")
    st.stop()

# =====================================================
# TABS
# =====================================================
tab_over, tab_prod, tab_cust, tab_country, tab_about = st.tabs([
    "📊 Overview",
    "🛍️ Products",
    "👥 Customers (RFM)",
    "🌍 Countries",
    "ℹ️ About Project"
])

# =====================================================
# TAB 1: OVERVIEW
# =====================================================
with tab_over:

    revenue = sales["Revenue"].sum()
    orders = sales["InvoiceNo"].nunique()
    customers = sales["CustomerID"].nunique()
    aov = revenue / orders
    cancelled_orders = canc["InvoiceNo"].nunique()
    cancel_rate = cancelled_orders / (orders + cancelled_orders) * 100

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("💰 Revenue", fmt_money(revenue))
    k2.metric("📦 Orders", f"{orders:,}")
    k3.metric("👤 Customers", f"{customers:,}")
    k4.metric("🧾 Avg Order Value", fmt_money(aov))
    k5.metric("❌ Cancellation Rate", f"{cancel_rate:.1f}%")

    st.divider()

    # ---------------- MONTH SPOTLIGHT (growth arrows) ----------------
    st.subheader("🔎 Month Spotlight")

    months = sorted(pd.Timestamp(m) for m in base_sales["Month"].unique())
    complete = [
        m for m in months
        if m + pd.offsets.MonthEnd(0) <= data_end.normalize()
    ]

    if complete:
        labels = {m.strftime("%B %Y"): m for m in complete}
        sel_label = st.selectbox(
            "Select a month (only complete months are shown)",
            list(labels.keys()),
            index=len(labels) - 1
        )
        sel_m = labels[sel_label]
        prev_m = sel_m - pd.DateOffset(months=1)

        cur = month_stats(base_sales, sel_m)
        prev = month_stats(base_sales, prev_m) if prev_m in months else None

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Revenue", fmt_money(cur["revenue"]),
                  delta=delta_str(cur["revenue"], prev["revenue"] if prev else None))
        s2.metric("Orders", f"{cur['orders']:,}",
                  delta=delta_str(cur["orders"], prev["orders"] if prev else None))
        s3.metric("Customers", f"{cur['customers']:,}",
                  delta=delta_str(cur["customers"], prev["customers"] if prev else None))
        s4.metric("Avg Order Value", fmt_money(cur["aov"]),
                  delta=delta_str(cur["aov"], prev["aov"] if prev else None))
    else:
        st.info("No complete month available for the selected country.")

    st.divider()

    # ---------------- TREND + MOM GROWTH ----------------
    monthly = sales.groupby("Month", as_index=False)["Revenue"].sum()
    monthly["MoM"] = monthly["Revenue"].pct_change() * 100

    t1, t2 = st.columns([2, 1])

    with t1:
        st.subheader("📈 Monthly Revenue Trend")
        fig_trend = px.line(monthly, x="Month", y="Revenue", markers=True)
        fig_trend.update_traces(line_color=ACCENT, fill="tozeroy",
                                mode="lines+markers")
        fig_trend.update_layout(xaxis_title="", yaxis_title="Revenue (£)")
        st.plotly_chart(style_fig(fig_trend), width="stretch")

    with t2:
        st.subheader("📊 Month-over-Month Growth")
        mom = monthly.dropna(subset=["MoM"]).copy()
        if mom.empty:
            st.info("Need at least 2 months of data.")
        else:
            mom["Direction"] = np.where(mom["MoM"] >= 0, "Growth", "Decline")
            fig_mom = px.bar(
                mom, x="Month", y="MoM", color="Direction",
                color_discrete_map={"Growth": GREEN, "Decline": RED}
            )
            fig_mom.update_layout(showlegend=False, xaxis_title="",
                                  yaxis_title="Growth (%)")
            st.plotly_chart(style_fig(fig_mom), width="stretch")

    st.caption("Note: data ends on 9 Dec 2011, so December 2011 is a partial month.")

    # ---------------- WEEKDAY + HOUR ----------------
    d1, d2 = st.columns(2)

    with d1:
        st.subheader("📅 Orders by Day of Week")
        by_day = (
            sales.groupby("Weekday")["InvoiceNo"].nunique()
            .reindex(WEEKDAYS, fill_value=0)
            .reset_index(name="Orders")
        )
        by_day["Highlight"] = np.where(
            by_day["Orders"] == by_day["Orders"].max(), "Busiest", "Other"
        )
        fig_day = px.bar(
            by_day, x="Weekday", y="Orders", color="Highlight",
            color_discrete_map={"Busiest": "#FF9F1C", "Other": "#4C78A8"}
        )
        fig_day.update_layout(showlegend=False, xaxis_title="")
        st.plotly_chart(style_fig(fig_day), width="stretch")

    with d2:
        st.subheader("🕐 Orders by Hour of Day")
        by_hour = (
            sales.groupby("Hour")["InvoiceNo"].nunique()
            .reset_index(name="Orders")
        )
        fig_hour = px.bar(
            by_hour, x="Hour", y="Orders",
            color="Orders", color_continuous_scale="Tealgrn"
        )
        fig_hour.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig_hour), width="stretch")

    st.divider()

    # ---------------- KEY INSIGHTS ----------------
    st.subheader("💡 Key Insights")

    peak = monthly.loc[monthly["Revenue"].idxmax()]

    prod_rev = sales.groupby("Description")["Revenue"].sum()
    best_product = prod_rev.idxmax()

    country_rev = sales.groupby("Country")["Revenue"].sum()
    top_country = country_rev.idxmax()
    country_share = country_rev.max() / revenue * 100

    busiest_day = by_day.loc[by_day["Orders"].idxmax()]

    cust_orders = (
        sales.dropna(subset=["CustomerID"])
        .groupby("CustomerID")["InvoiceNo"].nunique()
    )
    repeat_rate = (cust_orders > 1).mean() * 100 if len(cust_orders) else 0
    cancelled_value = abs(canc["Revenue"].sum())

    st.markdown(f"""
- **Peak month:** {peak['Month'].strftime('%B %Y')} with revenue of {fmt_money(peak['Revenue'])}.
- **Top product:** {best_product.title()} ({fmt_money(prod_rev.max())}).
- **Top country:** {top_country} contributes {country_share:.1f}% of revenue.
- **Busiest day:** {busiest_day['Weekday']} ({int(busiest_day['Orders']):,} orders).
- **Repeat customers:** {repeat_rate:.1f}% of identified customers ordered more than once.
- **Cancellations:** {cancel_rate:.1f}% of orders, worth {fmt_money(cancelled_value)}.
""")

# =====================================================
# TAB 2: PRODUCTS
# =====================================================
with tab_prod:

    st.subheader("🔥 Top 10 Products")

    rank_by = st.radio("Rank by", ["Revenue", "Quantity"], horizontal=True)

    top_products = (
        sales.groupby("Description", as_index=False)[rank_by].sum()
        .sort_values(rank_by, ascending=False)
        .head(10)
        .sort_values(rank_by)
    )
    fig_top = px.bar(
        top_products, x=rank_by, y="Description", orientation="h",
        color=rank_by, color_continuous_scale="Tealgrn"
    )
    fig_top.update_layout(coloraxis_showscale=False, yaxis_title="",
                          xaxis_title="Revenue (£)" if rank_by == "Revenue" else "Units sold")
    st.plotly_chart(style_fig(fig_top), width="stretch")

    st.divider()

    # ---------------- PRODUCT EXPLORER ----------------
    st.subheader("🔎 Product Explorer")

    ranked_products = prod_rev.sort_values(ascending=False).index.tolist()
    choice = st.selectbox(
        "Search or select a product (type to search)", ranked_products
    )

    p_sales = sales[sales["Description"] == choice]
    p_canc = canc[canc["Description"] == choice]

    p_revenue = p_sales["Revenue"].sum()
    p_units = p_sales["Quantity"].sum()
    p_orders = p_sales["InvoiceNo"].nunique()
    p_price = p_revenue / p_units if p_units else 0
    p_cancel_pct = abs(p_canc["Quantity"].sum()) / p_units * 100 if p_units else 0

    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("Revenue", fmt_money(p_revenue))
    q2.metric("Units Sold", f"{p_units:,}")
    q3.metric("Orders", f"{p_orders:,}")
    q4.metric("Avg Selling Price", f"£{p_price:,.2f}")
    q5.metric("Cancelled Units", f"{p_cancel_pct:.1f}%")

    e1, e2 = st.columns([2, 1])

    with e1:
        st.markdown("**Monthly revenue of this product**")
        p_month = p_sales.groupby("Month", as_index=False)["Revenue"].sum()
        fig_p = px.line(p_month, x="Month", y="Revenue", markers=True)
        fig_p.update_traces(line_color=ACCENT)
        fig_p.update_layout(xaxis_title="", yaxis_title="Revenue (£)")
        st.plotly_chart(style_fig(fig_p), width="stretch")

    with e2:
        st.markdown("**Top countries for this product**")
        p_country = (
            p_sales.groupby("Country", as_index=False)["Revenue"].sum()
            .sort_values("Revenue", ascending=False)
            .head(5)
            .sort_values("Revenue")
        )
        fig_pc = px.bar(p_country, x="Revenue", y="Country", orientation="h")
        fig_pc.update_traces(marker_color=ACCENT)
        fig_pc.update_layout(yaxis_title="", xaxis_title="Revenue (£)")
        st.plotly_chart(style_fig(fig_pc), width="stretch")

# =====================================================
# TAB 3: CUSTOMERS (RFM)
# =====================================================
with tab_cust:

    st.subheader("👥 Customer Segmentation (RFM Analysis)")
    st.caption(
        "RFM = Recency (how recently they bought), Frequency (how often), "
        "Monetary (how much they spent). Only customers with a known "
        "CustomerID are included."
    )

    cust = sales.dropna(subset=["CustomerID"])

    rfm = (
        cust.groupby("CustomerID")
        .agg(
            LastPurchase=("InvoiceDate", "max"),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("Revenue", "sum")
        )
        .reset_index()
    )

    if len(rfm) < 50:
        st.info("Not enough customers in this selection for RFM analysis. "
                "Widen the date range or country filter.")
    else:
        snapshot = fdf["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
        rfm["Recency"] = (snapshot - rfm["LastPurchase"].dt.normalize()).dt.days

        rfm["R"] = quantile_score(rfm["Recency"], reverse=True)
        rfm["M"] = quantile_score(rfm["Monetary"])
        rfm["F"] = pd.cut(
            rfm["Frequency"], bins=[0, 1, 2, 4, 7, np.inf],
            labels=[1, 2, 3, 4, 5]
        ).astype(int)

        conditions = [
            (rfm["R"] >= 4) & (rfm["F"] >= 4),
            (rfm["R"] >= 3) & (rfm["F"] >= 3),
            (rfm["R"] >= 4) & (rfm["F"] <= 2),
            (rfm["R"] <= 2) & (rfm["F"] >= 3),
            (rfm["R"] <= 2) & (rfm["F"] <= 2),
        ]
        names = ["Champions", "Loyal", "New Customers", "At Risk", "Lost"]
        rfm["Segment"] = np.select(conditions, names, default="Need Attention")

        seg = (
            rfm.groupby("Segment")
            .agg(
                Customers=("CustomerID", "count"),
                Revenue=("Monetary", "sum"),
                AvgRecencyDays=("Recency", "mean"),
                AvgOrders=("Frequency", "mean"),
                AvgSpend=("Monetary", "mean")
            )
            .reindex(SEGMENT_ORDER)
            .rename_axis("Segment")
            .dropna(subset=["Customers"])
            .reset_index()
        )
        seg["PctCustomers"] = seg["Customers"] / seg["Customers"].sum() * 100
        seg["PctRevenue"] = seg["Revenue"] / seg["Revenue"].sum() * 100

        champ = seg[seg["Segment"] == "Champions"]
        if not champ.empty:
            st.success(
                f"🏆 Champions are {champ['PctCustomers'].iloc[0]:.1f}% of "
                f"customers but bring {champ['PctRevenue'].iloc[0]:.1f}% of revenue."
            )

        r1, r2 = st.columns(2)

        with r1:
            st.markdown("**Customers per segment**")
            fig_seg = px.pie(
                seg, names="Segment", values="Customers", hole=0.5,
                color="Segment", color_discrete_map=SEGMENT_COLORS,
                category_orders={"Segment": SEGMENT_ORDER}
            )
            st.plotly_chart(style_fig(fig_seg), width="stretch")

        with r2:
            st.markdown("**Revenue per segment**")
            fig_rev = px.bar(
                seg, x="Segment", y="Revenue", color="Segment",
                color_discrete_map=SEGMENT_COLORS,
                category_orders={"Segment": SEGMENT_ORDER}
            )
            fig_rev.update_layout(showlegend=False, xaxis_title="",
                                  yaxis_title="Revenue (£)")
            st.plotly_chart(style_fig(fig_rev), width="stretch")

        st.markdown("**Segment summary**")
        seg_table = seg.rename(columns={
            "AvgRecencyDays": "Avg Days Since Last Order",
            "AvgOrders": "Avg Orders",
            "AvgSpend": "Avg Spend (£)",
            "PctCustomers": "% of Customers",
            "PctRevenue": "% of Revenue",
            "Revenue": "Revenue (£)"
        }).round(1)
        st.dataframe(seg_table, width="stretch", hide_index=True)

        with st.expander("What does each segment mean and what should the business do?"):
            st.markdown("""
- **Champions**: bought recently, buy often. Reward them (early access, loyalty perks).
- **Loyal**: regular buyers. Upsell and ask for reviews or referrals.
- **New Customers**: bought recently but only once or twice. Send a welcome offer to get a second order.
- **Need Attention**: average recency, low frequency. Remind them with a personal offer before they drift away.
- **At Risk**: used to buy often, but have not bought for a long time. Run a win-back campaign.
- **Lost**: low frequency and a long silence. Low priority, try a cheap re-activation email.
            """)

        st.divider()

        st.subheader("🏆 Top 10 Customers by Revenue")
        top_customers = (
            cust.groupby("CustomerID", as_index=False)
            .agg(
                country=("Country", "first"),
                orders=("InvoiceNo", "nunique"),
                revenue=("Revenue", "sum")
            )
            .sort_values("revenue", ascending=False)
            .head(10)
        )
        top_customers["revenue"] = top_customers["revenue"].round(0)
        st.dataframe(top_customers, width="stretch", hide_index=True)

        export = rfm[["CustomerID", "Recency", "Frequency", "Monetary",
                      "R", "F", "M", "Segment"]]
        st.download_button(
            "⬇ Download RFM table (CSV)",
            data=export.to_csv(index=False).encode("utf-8"),
            file_name="rfm_segments.csv",
            mime="text/csv"
        )

# =====================================================
# TAB 4: COUNTRIES
# =====================================================
with tab_country:

    st.subheader("🌍 Revenue by Country")

    by_country = (
        sales.groupby("Country")
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("InvoiceNo", "nunique"),
            Customers=("CustomerID", "nunique")
        )
        .reset_index()
    )
    by_country["AvgOrderValue"] = by_country["Revenue"] / by_country["Orders"]
    by_country["ShareOfRevenue"] = by_country["Revenue"] / by_country["Revenue"].sum() * 100

    excl_uk = st.checkbox(
        "Exclude United Kingdom from the map and chart "
        "(it is ~85% of revenue and hides the other countries)",
        value=True
    )

    shown = by_country[by_country["Country"] != "United Kingdom"] if excl_uk else by_country

    if shown.empty:
        st.info("No other countries in the current selection.")
    else:
        map_df = shown.copy()
        map_df["MapCountry"] = map_df["Country"].replace(
            {"EIRE": "Ireland", "RSA": "South Africa"}
        )

        fig_map = px.choropleth(
            map_df,
            locations="MapCountry",
            locationmode="country names",
            color="Revenue",
            hover_name="Country",
            hover_data={"MapCountry": False, "Revenue": ":,.0f", "Orders": True},
            color_continuous_scale="Tealgrn"
        )
        fig_map.update_geos(showframe=False, showcoastlines=False,
                            bgcolor="rgba(0,0,0,0)",
                            projection_type="natural earth")
        st.plotly_chart(style_fig(fig_map, height=450), width="stretch")

        st.subheader("📊 Top 10 Countries")
        top_c = shown.sort_values("Revenue", ascending=False).head(10).sort_values("Revenue")
        fig_c = px.bar(
            top_c, x="Revenue", y="Country", orientation="h",
            color="Revenue", color_continuous_scale="Tealgrn"
        )
        fig_c.update_layout(coloraxis_showscale=False, yaxis_title="",
                            xaxis_title="Revenue (£)")
        st.plotly_chart(style_fig(fig_c), width="stretch")

    st.subheader("📋 Country Details")
    country_table = (
        by_country.sort_values("Revenue", ascending=False)
        .rename(columns={
            "Revenue": "Revenue (£)",
            "AvgOrderValue": "Avg Order Value (£)",
            "ShareOfRevenue": "% of Revenue"
        })
        .round(1)
    )
    st.dataframe(country_table, width="stretch", hide_index=True)

# =====================================================
# TAB 5: ABOUT PROJECT
# =====================================================
with tab_about:

    st.subheader("ℹ️ About this project")

    st.markdown("""
**Goal:** understand what sells, where, when and to whom for a UK-based online gift retailer,
and how much revenue is lost to cancellations.

**Data:** public UCI / Kaggle *Online Retail* dataset. About 5.4 lakh real transactions,
Dec 2010 to Dec 2011. Each row is one product line on an invoice.

**What I did:**
1. **Cleaned** the raw data (duplicates, non-product rows, invalid prices and quantities, cancellations, missing customers).
2. Calculated **KPIs**: revenue, orders, customers, average order value, cancellation rate.
3. Built **trend and growth analysis**: monthly revenue, month-over-month growth, day and hour patterns.
4. Did **product analysis** with a searchable product explorer.
5. Did **RFM customer segmentation** (Champions, Loyal, New, Need Attention, At Risk, Lost).
6. Did **country analysis** with a world map.

**Tools:** Python, Pandas, NumPy, Plotly, Streamlit.
    """)

    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE) as f:
            s = json.load(f)

        st.subheader("🧹 Data Cleaning Summary")
        st.markdown(f"""
- **Raw rows:** {s['raw_rows']:,}
- **Duplicate rows removed:** {s['duplicates_removed']:,}
- **Non-product rows removed** (postage, bank charges, vouchers): {s['non_product_removed']:,}
- **Invalid rows removed** (zero/negative price or quantity, adjustments): {s['invalid_removed']:,}
- **Rows without description removed:** {s['no_description_removed']:,}
- **Rows with missing CustomerID** (kept for sales, excluded from customer metrics): {s['missing_customer_rows']:,}
- **Cancellations** flagged separately (invoice starts with 'C'): {s['cancellation_rows']:,} rows
- **Final clean rows:** {s['final_rows']:,} ({s['start_date']} to {s['end_date']})
        """)