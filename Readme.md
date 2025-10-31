
<p align="center">
  <img src="https://github.com/mallouliemna09-lang/Recherche-de-critiques-similaires/blob/main/Interface.png" alt="Aperçu de l'application Streamlit" width="90%">
</p>

# 🎬 Système de recommandation de critiques similaires

Une interface Streamlit interactive permettant de retrouver les critiques les plus similaires d’un même film, à partir d’un texte saisi par l’utilisateur.

---

## 🧭 Architecture du système

Ce diagramme illustre le flux complet du système : depuis le chargement des critiques jusqu’à la recherche de similarité dans Streamlit.

```mermaid
flowchart LR

    %% --- Sources de données ---
    subgraph DATA["📂 Données d'entrée"]
        A1[interstellar_critiques.csv]
        A2[fightclub_critiques.csv]
    end

    %% --- Pipeline offline ---
    subgraph PIPELINE["🛠 Prétraitement & Indexation"]
        B1[Nettoyage HTML strip_html()]
        B2[Concaténation titre + contenu → full_review]
        B3[Chunking 512 tokens + overlap]
        B4[Embedding des chunks MiniLM multilingue]
        B5[Construction index FAISS (IndexFlatIP)]
        B6[Sauvegarde des métadonnées meta_chunks_<film>.csv]
    end

    %% --- Artifacts par film ---
    subgraph ARTIFACTS["💾 Artifacts par film"]
        C1[faiss_Interstellar.index]
        C2[meta_chunks_Interstellar.csv]
        C3[faiss_FightClub.index]
        C4[meta_chunks_FightClub.csv]
    end

    %% --- Interface interactive ---
    subgraph APP["🌐 Application Streamlit"]
        D1[Choix du film selectbox("Interstellar"/"Fight Club")]
        D2[Saisie / Coller une critique]
        D3[Encodage de la requête MiniLM]
        D4[Recherche dans FAISSdu film choisi]
        D5[Regroupement par critique originale + tri par score]
        D6[Affichage des critiques similaires + score de similarité]
    end

    %% Flux de gauche à droite
    A1 --> B1
    A2 --> B1
    B1 --> B2 --> B3 --> B4 --> B5 --> ARTIFACTS
    B4 --> B6 --> ARTIFACTS

    %% Sélection dynamique à l'exécution
    ARTIFACTS -->|load_resources_for_movie(movie_name)| D1
    D1 --> D2 --> D3 --> D4 --> D5 --> D6

## 💡Démarche de conception

J’ai adopté une approche exploratoire et progressive pour concevoir un moteur de recherche sémantique robuste :

1. **Analyse du besoin produit :**  
   Je me suis d’abord concentrée sur la finalité fonctionnelle : proposer des critiques similaires d’un même film, non pas sur des mots-clés identiques, mais sur le sens global.

2. **Exploration des données :**  
   En examinant le CSV, j’ai constaté que les critiques étaient souvent longues, riches en HTML et comportaient à la fois un titre et un contenu textuel détaillé.
   J’ai donc décidé de fusionner les deux colonnes et de supprimer les balises HTML via BeautifulSoup pour obtenir un texte propre (full_review).

3. **Première approche :**  
   J’ai commencé par encoder directement chaque critique avec le modèle
   sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (déjà utilisé lors de mon précédent stage pour un moteur de recherche sémantique basé sur FAISS).
   Cette approche a confirmé la validité du modèle, mais j’ai rapidement identifié une limite.

4. **Problème rencontré :**  
   Certaines critiques dépassent la limite de **512 tokens**, entraînant une perte d’information.

5. **Amélioration :**  
   J’ai alors introduit une segmentation du texte (chunking) avec chevauchement (stride=400) afin de préserver la cohérence sémantique d’un texte long et éviter de couper une idée en deux. Un fichier `meta_chunks.csv` relie chaque chunk à sa critique d’origine.  

6. **Recherche améliorée :**  
   - Si la requête est longue, elle est également découpée en chunks  
   - Les résultats sont ensuite **agrégés et dédupliqués par critique originale**

7. **Résultats observés :**  
   - Copie exacte d’une critique → score ≈ **1.0**  
   - Reformulation / résumé → score ≈ **0.75 – 0.85**

8. **Adaptation multi-films:**

   L’énoncé imposant que les recommandations concernent uniquement le même film, j’ai adapté la logique d’indexation :
   J’ai remplacé la fonction build_vector_database() par une version générique paramétrée (build_vector_database_for_movie) capable de générer un index FAISS pour chaque film séparément à partir de son CSV (interstellar_critiques.csv et fightclub_critiques.csv).

   Chaque film dispose ainsi de ses propres fichiers :
      -faiss_interstellar.index, meta_chunks_interstellar.csv
      -faiss_fightclub.index, meta_chunks_fightclub.csv

   Cette séparation garantit que les critiques suggérées appartiennent toujours au même film que la requête.

   Dans l’application Streamlit, j’ai ajouté un sélecteur (movie_choice) permettant de choisir le film avant la recherche.
   Selon ce choix, le moteur charge automatiquement l’index FAISS correspondant.
---

## ⚙️ Choix techniques justifiés

| Composant | Choix | Justification |
|------------|--------|----------------|
| **Modèle d’embedding** | `paraphrase-multilingual-MiniLM-L12-v2` | Léger, performant, multilingue (français inclus). Déjà utilisé dans un projet précédent. |
| **Moteur de recherche** | **FAISS (IndexFlatIP)** | Recherche vectorielle rapide sur CPU et open-source Produit scalaire ≈ similarité cosinus. |
| **Normalisation** | `normalize_embeddings=True` | Permet de comparer les vecteurs sur une même échelle. |
| **Interface utilisateur** | `Streamlit` | Permet de créer rapidement une interface web interactive. Déjà utilisé dans mon dernier stage. |

---

## 📓 Notebook d’exploration

J’ai inclus un **notebook Jupyter (`exploration.ipynb`)** retraçant toutes les étapes de conception :

1. **Inspection et nettoyage :**  
    -Fusion review_title + review_content
    -Suppression des balises HTML via BeautifulSoup
    -Création d’une colonne full_review pour les textes lisibles
2. **Analyse de la longueur :**  
   -Comptage des tokens avec le tokenizer HuggingFace
   -Constat : plusieurs critiques dépassent 512 tokens

3. **Première approche :**  
   -Encodage direct avec MiniLM-L12-v2
   -Indexation FAISS simple 

4. **Problème identifié :**  
   Troncature des textes longs → perte de sens.  

5. **Amélioration :**  
   -Introduction du chunking avec overlap
   -Indexation de chaque chunk dans FAISS
   -Sauvegarde des métadonnées (meta_chunks.csv)

6. **Recherche finale :**  
   -Version simple : requête encodée en un seul vecteur
   -Version améliorée : requête segmentée en chunks et agrégation des résultats

7. **Résultats :**  
   - Texte identique → score 1.0  
   - Reformulation → critiques similaires retrouvées (score ≈ 0.8)  


🚀 Exécution
1️⃣ Installer les dépendances
```powershell
     pip install -r requirements.txt
```
2️⃣ Lancer l’application Streamlit
```powershell
     streamlit run app.py
```

