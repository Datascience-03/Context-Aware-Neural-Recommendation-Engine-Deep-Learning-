import json
import os
import urllib.request
import streamlit as st

st.set_page_config(
    page_title="Neural Recommendation Engine",
    page_icon="🛍️",
    layout="wide",
)

st.title("🛍️ Context-Aware Neural Recommendation Engine")
st.markdown(
    "Interactive deep learning **Two-Tower Neural Retrieval Service** predicting personalized fashion products."
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
CATALOG_PATH = "data/processed/articles_catalog.json"

@st.cache_data
def load_catalog():
    if os.path.exists(CATALOG_PATH):
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

catalog = load_catalog()

# Sidebar: System Health & Info
with st.sidebar:
    st.header("⚙️ Backend Engine Status")
    try:
        health_resp = urllib.request.urlopen(f"{API_BASE_URL}/health", timeout=3)
        health_data = json.loads(health_resp.read().decode())
        st.success("● Inference Server Live")
        st.metric("Indexed Products", health_data.get("num_items", 0))
        st.metric("Active Customer Profiles", health_data.get("num_users", 0))
        st.metric("Embedding Dimension", f"{health_data.get('embedding_dim', 64)}D")
        st.metric("Vector Index", health_data.get("index_type", "ExactSearchIndex"))
    except Exception as e:
        st.error(f"● Backend Unreachable ({e})")
        st.info("Make sure the backend is running on http://127.0.0.1:8000")

    st.markdown("---")
    st.markdown("**Sample Customer IDs to try:**")
    st.markdown("• `1`, `2`, `3`, `10`, `42` *(Known customers with unique personalized models)*")
    st.markdown("• `9999`, `new_visitor` *(Cold-start / brand new customer fallback)*")

# Main Section: User & Context Inputs
st.subheader("1. Configure Customer & Real-Time Shopping Context")

# Quick scenario presets
preset_cols = st.columns(4)
with preset_cols[0]:
    if st.button("❄️ Winter Weekend Online", use_container_width=True):
        st.session_state["season_idx"] = 1
        st.session_state["weekend_val"] = 1
        st.session_state["channel_val"] = 2
with preset_cols[1]:
    if st.button("☀️ Summer Weekend Store", use_container_width=True):
        st.session_state["season_idx"] = 0
        st.session_state["weekend_val"] = 1
        st.session_state["channel_val"] = 1
with preset_cols[2]:
    if st.button("🍂 Autumn Workday Online", use_container_width=True):
        st.session_state["season_idx"] = 3
        st.session_state["weekend_val"] = 0
        st.session_state["channel_val"] = 2
with preset_cols[3]:
    if st.button("🌸 Spring Weekday Store", use_container_width=True):
        st.session_state["season_idx"] = 2
        st.session_state["weekend_val"] = 0
        st.session_state["channel_val"] = 1

col1, col2, col3 = st.columns(3)

with col1:
    customer_id = st.text_input(
        "Customer ID",
        value="1",
        help="Enter any trained ID (1 to 300) for personalized recommendations, or an unseen ID for cold-start fallback."
    )
    top_k = st.slider("Number of Recommendations (Top-K)", min_value=1, max_value=20, value=6)

with col2:
    season_idx = st.session_state.get("season_idx", 0)
    season = st.selectbox("Current Season", ["Summer", "Winter", "Spring", "Autumn"], index=season_idx)
    channel_val = st.session_state.get("channel_val", 2)
    sales_channel = st.selectbox("Sales Channel", [1, 2], index=0 if channel_val == 1 else 1, format_func=lambda x: f"Channel {x} ({'Online' if x == 2 else 'Physical Store'})")

with col3:
    weekend_val = st.session_state.get("weekend_val", 1)
    is_weekend = st.radio("Shopping Context", [0, 1], index=weekend_val, format_func=lambda x: "Weekend" if x == 1 else "Weekday", horizontal=True)

st.markdown("---")

# Recommend Button & Results
if st.button("🚀 Generate Neural Recommendations", type="primary"):
    payload = {
        "customer_id": str(customer_id).strip(),
        "top_k": int(top_k),
        "context": {
            "season": season,
            "is_weekend": int(is_weekend),
            "sales_channel_id": int(sales_channel),
        },
    }

    try:
        req = urllib.request.Request(
            f"{API_BASE_URL}/recommend",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with st.spinner("Embedding query & searching vector index..."):
            resp = urllib.request.urlopen(req, timeout=5)
            data = json.loads(resp.read().decode())

        st.subheader("2. Recommended Products")
        meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
        meta_col1.metric("Retrieval Latency", f"{data.get('retrieval_time_ms', 0):.2f} ms")
        is_cold = data.get("cold_start", False)
        meta_col2.metric("User Profile Mode", "Cold-Start Fallback" if is_cold else "Personalized Model")
        meta_col3.metric("Cold Start User?", "Yes" if is_cold else "No")
        meta_col4.metric("Cache Hit?", "Yes" if data.get("cache_hit") else "No")

        recs = data.get("recommendations", [])
        if not recs:
            st.warning("No recommendations returned.")
        else:
            # Display items in a responsive 3-column card grid
            cols = st.columns(3)
            for i, rec in enumerate(recs):
                art_id = str(rec["article_id"])
                product = catalog.get(art_id, {
                    "name": f"Fashion Article #{art_id}",
                    "category": "Apparel",
                    "color": "Standard",
                    "price": "$49.99",
                    "image_url": "https://images.unsplash.com/photo-1521572267360-ee0c2909d518?auto=format&fit=crop&w=400&q=80",
                    "rating": 4.5,
                    "reviews_count": 48
                })

                with cols[i % 3]:
                    st.markdown(
                        f"""
                        <div style="border: 1px solid #1e293b; border-radius: 12px; overflow: hidden; background: #0f172a; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.25);">
                            <div style="position: relative; height: 220px; overflow: hidden;">
                                <img src="{product['image_url']}" style="width: 100%; height: 100%; object-fit: cover; display: block;" />
                                <span style="position: absolute; top: 10px; left: 10px; background: rgba(14, 165, 233, 0.9); color: white; padding: 3px 8px; border-radius: 6px; font-size: 11px; font-weight: bold;">
                                    RANK #{rec['rank']}
                                </span>
                                <span style="position: absolute; top: 10px; right: 10px; background: rgba(15, 23, 42, 0.85); color: #38bdf8; padding: 3px 8px; border-radius: 6px; font-size: 11px; font-weight: 600;">
                                    Match: {rec['score']:.4f}
                                </span>
                            </div>
                            <div style="padding: 14px;">
                                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;">
                                    <span style="font-size: 11px; text-transform: uppercase; color: #94a3b8; font-weight: 600;">{product['category']} • {product['color']}</span>
                                    <span style="font-size: 16px; font-weight: bold; color: #10b981;">{product['price']}</span>
                                </div>
                                <h4 style="margin: 0 0 8px 0; font-size: 15px; color: #f8fafc; line-height: 1.3;">{product['name']}</h4>
                                <div style="display: flex; justify-content: space-between; align-items: center; font-size: 12px; color: #64748b; border-top: 1px solid #1e293b; padding-top: 8px;">
                                    <span>★ {product['rating']} ({product['reviews_count']} reviews)</span>
                                    <span style="font-family: monospace; color: #94a3b8;">ID: #{art_id}</span>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    except Exception as e:
        st.error(f"Failed to fetch recommendations: {e}")
