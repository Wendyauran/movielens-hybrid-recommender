import streamlit as st

st.set_page_config(
    page_title="Movie Recommendation System", 
    page_icon="🎬", 
    layout="wide"
)

pages = [
    st.Page(
        "pages/1_overview.py",
        title="Home",
        icon="🏠"
    ),
    st.Page(
        "pages/2_recommendations.py",
        title="Recommendations",
        icon="🎬"
    ),
    st.Page(
        "pages/3_movie_explorer.py",
        title="Movie Explorer",
        icon="🔍"
    ),
    st.Page(
        "pages/4_analytics.py",
        title="Analytics",
        icon="📊"
    ),
    st.Page(
        "pages/5_about.py",
        title="About",
        icon="ℹ️"
    )
]

pg = st.navigation(pages)
pg.run()