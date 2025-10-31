import streamlit as st
from search_engine import load_resources_for_movie, search_similarity_3


@st.cache_resource
def get_resources(movie_name):
    return load_resources_for_movie(movie_name)



st.set_page_config(page_title="Recherche de critiques similaires")
st.title(" 🎬 Recherche de critiques similaires")
movie_choice = st.selectbox("Choisir un film :", ["Interstellar", "Fight Club"])
user_text = st.text_area("Colle ta critique :", height=250)
k = st.slider("Nombre de recommandations :", 1, 10, 5)
page_element="""
<style>
[data-testid="stAppViewContainer"]{
background-color: #b7d3d6;
  background-image: url("https://img.freepik.com/photos-gratuite/billets-cinema-popcorns-fond-bleu_23-2148188223.jpg?semt=ais_hybrid&w=740&q=80");
  background-size: 40%;
  background-repeat: no-repeat;
  background-position: right center;
}
[data-testid="stHeader"]{
  background-color: rgba(0,0,0,0);
}
</style>
"""
st.markdown(page_element, unsafe_allow_html=True)

if st.button("Trouver des critiques proches:"):
    if user_text.strip():
        with st.spinner("Recherche en cours..."):
                movie_name=movie_choice
                chunks_df, index, model, tokenizer = get_resources(movie_name)
                df_resultats = search_similarity_3(
                    query_text=user_text,
                    k=k,
                    chunks_df=chunks_df,
                    index=index,
                    model=model,
                    tokenizer=tokenizer
                )

        
        
        st.subheader("Critiques similaires :")
        for _, row in df_resultats.iterrows():
            st.markdown(f"**Auteur :** {row['username']}")
            st.write(row["full_review"]+ "...")
            st.caption(f"Score de similarité : {row['score']:.3f}")
            st.divider()


