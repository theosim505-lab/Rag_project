# 🍿 CinéBot RAG

**CinéBot RAG** est un assistant intelligent de recommandation de films basé sur une architecture **RAG (Retrieval-Augmented Generation)**. Il utilise **Streamlit** pour l'interface conversationnelle, **PostgreSQL (`pgvector`)** pour la recherche sémantique vectorielle dans un catalogue de films, et **Ollama** (`nomic-embed-text` + `llama3.2`) pour générer les embeddings et formuler des recommandations personnalisées avec affiches, synopsis et bandes-annonces.

## 🚀 Utilisation

### 1. Charger les films dans la base de données
Exécutez le script d'ingestion pour vectoriser le catalogue (`films.json`) et insérer les données et métadonnées dans PostgreSQL :

```bash
python ingest.py
```

### 2. Lancer l'application Streamlit
Démarrez ensuite l'interface web de CinéBot :

```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur (généralement à l'adresse `http://localhost:8501`).