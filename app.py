import feedparser
import random
import re
import warnings

# Importations Flask et Data Processing
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd

# Importations pour le Traitement du Langage Naturel (NLP)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords 

# --- Configuration et Initialisation ---

# Supprimer les avertissements de sklearn/pandas pour le mode développement
warnings.filterwarnings("ignore")

app = Flask(__name__)

# La clé de la Sérendipité : forcer la diversité des domaines pour casser la bulle.
FLUX_RSS_DIVERS = {
    'TechCrunch (Tech)': 'http://feeds.feedburner.com/TechCrunch/startups',
    'LeMonde (Planète/Science)': 'http://www.lemonde.fr/planete/rss_full.xml',
    'CourrierInt (Histoire/Culture)': 'https://www.courrierinternational.com/feed/rss/toute-lactualite/histoire',
    'Dezeen (Design/Art)': 'https://www.dezeen.com/feed/',
    'LesEchos (Économie)': 'https://www.lesechos.fr/rss/lesechos_economie.xml'
}

# Variable globale pour stocker tous les articles (simule une BDD pour le POC)
articles_df = pd.DataFrame()

# --- 1. Récupération et Nettoyage des Articles ---

def get_all_articles():
    """Récupère et normalise tous les articles des flux RSS dans un DataFrame."""
    global articles_df
    all_entries = []
    
    print("-> Chargement des flux RSS...")

    for source_name, url_flux in FLUX_RSS_DIVERS.items():
        try:
            flux = feedparser.parse(url_flux)
            for entry in flux.entries:
                
                # Le contenu pour le TF-IDF est la combinaison Titre + Résumé
                texte_complet = f"{entry.get('title', '')} {entry.get('summary', '')}"
                
                # S'assurer d'avoir un lien ou un titre pour l'ID
                link = entry.get('link', entry.get('title'))
                if not link:
                    continue
                
                all_entries.append({
                    'id': hash(link), 
                    'titre': entry.get('title', 'Titre non disponible'),
                    'lien': link,
                    'source': source_name,
                    'resume': entry.get('summary', 'Pas de résumé disponible'),
                    'texte_complet': re.sub(r'<.*?>', '', texte_complet) # Nettoyage HTML minimal
                })
        except Exception as e:
            print(f"Erreur lors du chargement du flux {source_name}: {e}")
            
    articles_df = pd.DataFrame(all_entries).drop_duplicates(subset=['lien']).reset_index(drop=True)
    print(f"-> {len(articles_df)} articles chargés.")
    return articles_df

# --- 2. Logique de Sérendipité (Reranking basé sur la distance sémantique) ---

def find_serendipitous_recommendations(article_id, top_n=3):
    """
    Trouve des recommandations en maximisant la similarité (pertinence)
    et en favorisant les sources différentes (diversité/nouveauté).
    """
    
    if articles_df.empty:
        return []

    article_id = int(article_id)

    # Article de référence choisi par l'utilisateur
    article_reference = articles_df[articles_df['id'] == article_id]
    if article_reference.empty:
        return []
        
    reference_index = article_reference.index[0]
    reference_source = article_reference['source'].iloc[0]
    
    # --- CORRECTION DE L'ERREUR stop_words ---
    try:
        # Tente de charger les stop words français depuis NLTK
        stop_words_fr = list(stopwords.words('french'))
    except LookupError:
        # Si NLTK n'est pas configuré, utilise 'english' comme plan de secours 
        # ou une liste vide si la majorité du contenu est français
        print("Avertissement: Les stop words français n'ont pas été trouvés. Utilisation d'une liste vide.")
        stop_words_fr = None 

    # 1. Vectorisation TF-IDF de tous les textes
    # N-grammes (1,2) : capture des paires de mots améliorant la sémantique.
    tfidf = TfidfVectorizer(stop_words=stop_words_fr, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(articles_df['texte_complet'])
    
    # 2. Calcul de la similarité Cosinus (Angle entre le vecteur choisi et les autres)
    cosine_sim = cosine_similarity(tfidf_matrix[reference_index], tfidf_matrix)
    
    sim_scores = pd.Series(cosine_sim[0], index=articles_df.index)
    
    # 3. Exclure l'article lui-même (score de 1.0)
    sim_scores = sim_scores.drop(reference_index, errors='ignore')
    
    # Trie par score de similarité (du plus similaire au moins similaire)
    sim_scores_sorted = sim_scores.sort_values(ascending=False)
    
    recommendations = []
    
    # 4. Reranking pour la Sérendipité (Pertinence élevée + Diversité assurée)
    
    for index, score in sim_scores_sorted.items():
        article_candidat = articles_df.loc[index]
        
        # Le critère Sérendipité/Diversité : source différente de l'article de référence.
        if article_candidat['source'] != reference_source:
            recommendations.append(article_candidat.to_dict())
            if len(recommendations) >= top_n:
                break
                
    # Plan de secours : si la diversité empêche de trouver suffisamment d'articles
    if len(recommendations) < top_n:
        remaining = top_n - len(recommendations)
        
        # On reprend les meilleurs scores parmi ceux qui n'ont pas encore été recommandés
        for index in sim_scores_sorted.index:
             article_candidat = articles_df.loc[index].to_dict()
             if article_candidat not in recommendations:
                 recommendations.append(article_candidat)
             if len(recommendations) >= top_n:
                break
                
    return recommendations


# --- Routes Flask ---

@app.before_request
def setup_articles():
    """Initialise le catalogue d'articles avant chaque requête si nécessaire."""
    global articles_df
    # Tente de recharger les articles s'il n'y en a pas (utile pour le premier lancement)
    if articles_df.empty or len(articles_df) < 5: 
        get_all_articles()

@app.route('/')
def home():
    """Phase 1: Affichage initial des 5 articles aléatoires et divers."""
    if articles_df.empty:
        return "Erreur de chargement des flux RSS. Veuillez réessayer.", 500

    # Sélection aléatoire de 5 articles
    initial_choices = articles_df.sample(n=min(5, len(articles_df)), random_state=random.randint(0, 10000)).to_dict('records')
    
    return render_template('results.html', 
                           stage='initial', 
                           articles=initial_choices,
                           titre_page="Sélectionnez votre article de découverte")

@app.route('/choose', methods=['POST'])
def choose():
    """
    Phase 2: L'utilisateur a choisi un article (POST). 
    On calcule et affiche les recommandations de sérendipité.
    """
    chosen_id = request.form.get('article_id')
    
    if not chosen_id:
        return redirect(url_for('home'))

    # L'ID est l'ancrage de pertinence pour la recherche
    recommendations = find_serendipitous_recommendations(chosen_id, top_n=3)
    
    # Récupérer l'article choisi pour l'affichage de contexte
    article_choisi = articles_df[articles_df['id'] == int(chosen_id)].iloc[0].to_dict()

    return render_template('results.html', 
                           stage='serendipity', 
                           articles=recommendations,
                           article_choisi=article_choisi,
                           titre_page="Vos recommandations de Sérendipité")


if __name__ == '__main__':
    print("--- Lancement du POC de Sérendipité ---")
    app.run(debug=True)