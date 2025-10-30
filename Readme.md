# 🎬 Système de recommandation de critiques similaires 

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

