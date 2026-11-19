import json
import os
import ollama
import psycopg
from psycopg.types.json import Json

# Paramètres de connexion (alignés sur votre fichier .env)
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "rag"
DB_USER = "rag_user"
DB_PASSWORD = "rag"

def main():
    # 1. Charger le fichier JSON
    if not os.path.exists("films.json"):
        print("Erreur : Le fichier films.json est introuvable.")
        return
        
    with open("films.json", "r", encoding="utf-8") as f:
        films = json.load(f)

    # 2. Connexion à PostgreSQL
    conn_string = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}"
    
    with psycopg.connect(conn_string) as conn:
        with conn.cursor() as cur:
            # Idempotence : Nettoyage de la table avant réinsertion
            print("Nettoyage de la table documents...")
            cur.execute("TRUNCATE TABLE documents RESTART IDENTITY;")
            
            print(f"Début de l'ingestion de {len(films)} fiches...")
            for film in films:
                # Composer un texte descriptif clair regroupant titre, genre et résumé
                combined_text = f"Titre : {film['title']}. Genre : {film['genre']}. Description : {film['description']}"
                
                # Générer le vecteur via nomic-embed-text (dimension 768)
                response = ollama.embeddings(
                    model='nomic-embed-text',
                    prompt=combined_text
                )
                embedding = response['embedding']
                
                # Préparer les métadonnées (genre, année)
                metadata = {
                    "genre": film["genre"],
                    "year": film["year"]
                }
                
                # Insertion dans la base de données
                cur.execute(
                    """
                    INSERT INTO documents (source, content, metadata, embedding)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        film["title"],
                        combined_text,
                        Json(metadata),
                        embedding
                    )
                )
            conn.commit()
    print("Ingestion terminée avec succès !")

if __name__ == "__main__":
    main()