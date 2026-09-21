import streamlit as st

st.set_page_config(
    page_title="Movie Recommendation System", 
    page_icon="🎬", 
    layout="wide"
)

st.markdown("""
<style>
[data-testid="stBaseButton-primary"] p {
    color: #18181B !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

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

pg = st.navigation({"Menu": pages})
pg.run()