import os
import streamlit as st
import ollama
import psycopg
from dotenv import load_dotenv

# Charger les variables du fichier .env
load_dotenv()

# Configuration de la page Streamlit (en mode wide pour un design moderne)
st.set_page_config(
    page_title="CinéBot - Assistant RAG",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Récupération sécurisée des paramètres depuis le .env
DB_HOST = "localhost"
DB_PORT = int(os.getenv("POSTGRES_PORT", 5432))
DB_NAME = os.getenv("POSTGRES_DB", "rag")
DB_USER = os.getenv("POSTGRES_USER", "rag_user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

# --- STYLE CSS PERSONNALISÉ POUR UN DESIGN PLUS MODERNE ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stChatInputContainer {
        padding-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# En-tête de l'application
st.title("🍿 CinéBot RAG")
st.markdown("Votre assistant intelligent propulsé par l'IA et PostgreSQL pour trouver le film parfait.")

# Barre latérale (Sidebar) épurée
with st.sidebar:
    st.header("⚙️ Paramètres")
    seuil_input = st.slider("Seuil de similarité minimal (%)", 0.0, 100.0, 50.0, 5.0)
    st.info("💡 **Astuce :** Essayez des requêtes comme *« Un film sombre sur les rêves »* ou *« Je veux voir un Spider-Man »*.")
    
    if st.button("🗑️ Effacer l'historique"):
        st.session_state.messages = []
        st.rerun()

# Initialisation de l'historique de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage de l'historique des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrée utilisateur via le chat
if user_query := st.chat_input("Que souhaitez-vous regarder aujourd'hui ?"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyse et recherche sémantique en cours..."):
            try:
                # 1. Générer l'embedding de la requête utilisateur
                embed_response = ollama.embeddings(
                    model='nomic-embed-text',
                    prompt=user_query
                )
                query_embedding = embed_response['embedding']

                # 2. Interroger PostgreSQL avec la distance cosinus
                conn_string = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}"
                candidates = []
                
                with psycopg.connect(conn_string) as conn:
                    with conn.cursor() as cur:
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
                            distance = float(row[3])
                            similarity_pct = max(0.0, (1.0 - distance)) * 100
                            candidates.append({
                                "title": row[0],
                                "content": row[1],
                                "metadata": row[2] if isinstance(row[2], dict) else {},
                                "similarity": similarity_pct
                            })

                # 3. Filtrer avec le seuil défini dans la sidebar
                valid_candidates = [c for c in candidates if c["similarity"] >= seuil_input]

                if not valid_candidates:
                    bot_reply = "❌ Je suis désolé, mais je n'ai trouvé aucun film dans mon catalogue qui corresponde à cette demande."
                    st.markdown(bot_reply)
                else:
                    candidates_text = ""
                    for idx, c in enumerate(valid_candidates, 1):
                        candidates_text += f"Candidat {idx}:\n- Titre : {c['title']}\n- Similarité : {c['similarity']:.1f}%\n- Détails : {c['content']}\n\n"

                    prompt_llm = f"""
Tu es un assistant de recommandation de films strict et logique.
Voici la demande de l'utilisateur : "{user_query}"

Voici les candidats extraits de notre base de données :
{candidates_text}

Consignes strictes :
1. Si les candidats n'ont aucun lien direct, réponds simplement : "Je ne sais pas ou je n'ai aucun film correspondant dans mon catalogue."
2. Sélectionne et présente au maximum 3 recommandations parmi cette liste de manière claire, moderne et élégante.
3. Mentionne le pourcentage de correspondance.
"""

                    llm_response = ollama.chat(
                        model='llama3.2',
                        messages=[{'role': 'user', 'content': prompt_llm}]
                    )
                    bot_reply = llm_response['message']['content']
                    st.markdown(bot_reply)

                    # --- AFFICHAGE MODERNE DES FILMS (Affiches + Bandes-annonces) ---
                    st.markdown("### 🎥 Films recommandés dans notre catalogue")
                    cols = st.columns(len(valid_candidates[:3]))
                    
                    for i, c in enumerate(valid_candidates[:3]):
                        meta = c["metadata"]
                        with cols[i]:
                            st.markdown(f"**{c['title']}**")
                            if "year" in meta:
                                st.caption(f"{meta.get('genre', 'Film')} • {meta.get('year', '')}")
                            
                            # Affichage de l'affiche si l'URL existe
                            poster = meta.get("poster_url")
                            if poster:
                                st.image(poster, use_container_width=True)
                            
                            # Jauge de correspondance
                            st.progress(int(c["similarity"]))
                            st.caption(f"Correspondance : **{c['similarity']:.1f}%**")
                            
                            # Lien bande-annonce
                            trailer = meta.get("trailer_url")
                            if trailer:
                                st.markdown(f"[▶️ Voir la bande-annonce]({trailer})")

                # Détails techniques dans un expander (Debug)
                with st.expander("🔍 Analyse vectorielle détaillée (Debug)"):
                    for c in candidates:
                        status = "✅ Valide" if c["similarity"] >= seuil_input else "❌ Rejeté"
                        st.markdown(f"**{c['title']}** — {status} ({c['similarity']:.1f}%)")
                        st.progress(int(c['similarity']))

                # Enregistrer la réponse dans l'historique
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})

            except Exception as e:
                st.error(f"Une erreur est survenue : {str(e)}")