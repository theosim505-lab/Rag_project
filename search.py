import sys
import ollama
import psycopg

DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "rag"
DB_USER = "rag_user"
DB_PASSWORD = "rag"

def search_recommendations(user_query: str):
    print(f"\n Requête utilisateur : « {user_query} »\n")
    
    # 1. Transformer la requête en vecteur via nomic-embed-text
    response = ollama.embeddings(
        model='nomic-embed-text',
        prompt=user_query
    )
    query_embedding = response['embedding']
    
    # 2. Exécuter la requête SQL avec la distance cosinus (<=>) pour les 5 plus proches
    conn_string = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}"
    candidates = []
    
    with psycopg.connect(conn_string) as conn:
        with conn.cursor() as cur:
            # L'opérateur <=> calcule la distance cosinus
            cur.execute(
                """
                SELECT source, content, metadata, (embedding <=> %s::vector) AS distance
                FROM documents
                ORDER BY distance ASC
                LIMIT 5;
                """,
                (query_embedding,)
            )
            rows = cur.fetchall()
            for row in rows:
                candidates.append({
                    "title": row[0],
                    "content": row[1],
                    "metadata": row[2],
                    "distance": float(row[3])
                })
                
    if not candidates:
        print("Aucun élément trouvé dans la base de données.")
        return

    # 3. Préparer le contexte pour Llama 3.2
    candidates_text = ""
    for idx, c in enumerate(candidates, 1):
        candidates_text += f"Candidat {idx}:\n- Titre : {c['title']}\n- Détails : {c['content']}\n\n"

    prompt_llm = f"""
Tu es un assistant de recommandation de films intelligent et rigoureux.
Voici une demande d'un utilisateur : "{user_query}"

Voici les 5 meilleurs candidats extraits de notre catalogue via recherche sémantique :
{candidates_text}

Consignes strictes :
1. Sélectionne et présente les 3 meilleures recommandations parmi ces candidats.
2. Justifie en exactement deux phrases pourquoi chaque choix correspond à la demande de l'utilisateur, en t'appuyant uniquement sur les descriptions fournies.
3. Si aucun élément du catalogue ne correspond réellement à la demande, signale-le clairement.
"""

    print("Interrogation de model='llama3.2', pour la sélection finale...\n")
    llm_response = ollama.chat(
        model='llama3.2',
        messages=[{'role': 'user', 'content': prompt_llm}]
    )
    
    print("--- RÉSULTATS DE LA RECOMMANDATION ---")
    print(llm_response['message']['content'])

if __name__ == "__main__":
    # Vous pouvez tester avec une requête de votre choix
    query = "Un film de science-fiction sombre avec une intrigue autour du temps ou de l'espace"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    search_recommendations(query)