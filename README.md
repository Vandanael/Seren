-----

# README: Projet POC de Sérendipité

## 1\. Objectif du Projet

Ce projet est une **Preuve de Concept (POC)** d'un moteur de recommandation axé sur les objectifs **"Beyond Accuracy"** (au-delà de la précision), notamment la **Sérendipité**, la **Diversité**, et la **Nouveauté**. Il vise à démontrer comment une mesure de **proximité sémantique** peut être utilisée pour générer des recommandations pertinentes, mais inattendues.

-----

## 2\. Fonctionnement Algorithmique

Le processus est structuré en deux phases intermédiaires pour contrôler le niveau de surprise et de pertinence :

### 2.1. Phase d'Initialisation (Diversité)

L'application agrège des contenus de **flux RSS thématiquement éloignés**. Elle présente à l'utilisateur une sélection initiale d'articles aléatoires, forçant un signal de préférence sur un sujet a priori **non habituel**.

### 2.2. Phase de Reranking Sémantique

1.  **Vectorisation :** Le contenu textuel des articles est converti en vecteurs numériques pondérés à l'aide de l'algorithme **TF-IDF** (Term Frequency-Inverse Document Frequency).
2.  **Mesure de Pertinence :** La **Similarité Cosinus** est calculée entre l'article sélectionné (l'ancrage) et tous les autres articles du catalogue.
3.  **Contrôle de Sérendipité :** Le système effectue un **Reranking** en privilégiant les articles présentant une **haute similarité sémantique** (pertinence) mais provenant impérativement d'une **source RSS différente** (diversité et nouveauté).

-----

## 3\. Mise en Route

### 3.1. Prérequis

Python 3.x.

### 3.2. Installation des Dépendances

Dans le répertoire du projet, exécutez l'installation des librairies requises :

```bash
python -m pip install Flask feedparser scikit-learn pandas nltk
```

Téléchargez ensuite les mots vides français pour NLTK :

```bash
python
>>> import nltk
>>> nltk.download('stopwords')
>>> exit()
```

### 3.3. Lancement

Lancez l'application en mode développement :

```bash
python app.py
```

Accédez à l'application via votre navigateur à l'adresse `http://127.0.0.1:5000/`.

-----

## 4\. Architecture

| Fichier | Rôle |
| :--- | :--- |
| `app.py` | Logique d'application, agrégation RSS, implémentation du **TF-IDF/Similarité Cosinus**, et algorithme de reranking. |
| `templates/results.html` | Interface utilisateur gérant l'affichage de la phase de choix initiale et des recommandations finales. |
