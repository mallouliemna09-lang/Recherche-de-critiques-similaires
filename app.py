import streamlit as st
from search_engine import load_resources, search_similarity_3


@st.cache_resource
def get_resources():
    return load_resources()



st.set_page_config(page_title="Recherche de critiques similaires")
st.title(" 🎬 Recherche de critiques similaires")
user_text = st.text_area("Colle ta critique :", height=250)
k = st.slider("Nombre de recommandations :", 1, 10, 5)


if st.button("Trouver des critiques proches:"):
    if user_text.strip():
        with st.spinner("Recherche en cours..."):
                chunks_df, index, model, tokenizer = get_resources()
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


