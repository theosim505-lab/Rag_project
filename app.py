import os
import streamlit as st
import ollama
import psycopg
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="CinéBot - Assistant RAG",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paramètres BDD (.env)
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = int(os.getenv("POSTGRES_PORT", 5432))
DB_NAME = os.getenv("POSTGRES_DB", "rag")
DB_USER = os.getenv("POSTGRES_USER", "rag_user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
# Optionnel : lien vers votre interface BDD (pgAdmin, Adminer, etc.) ou URI de connexion
DB_ADMIN_URL = os.getenv("DB_ADMIN_URL", f"postgresql://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

conn_string = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}"

def get_db_stats():
    try:
        with psycopg.connect(conn_string, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM documents;")
                count = cur.fetchone()[0]
        return True, count, None
    except Exception as exc:
        return False, 0, str(exc)

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

st.title("🍿 CinéBot RAG")
st.markdown("Votre assistant intelligent propulsé par l'IA et PostgreSQL pour trouver le film parfait.")

db_ok, doc_count, db_err = get_db_stats()

with st.sidebar:
    st.header("⚙️ Paramètres")
    seuil_input = st.slider("Seuil de similarité minimal (%)", 0.0, 100.0, 50.0, 5.0)
    st.info("💡 **Astuce :** Essayez des requêtes comme *« Un film sombre sur les rêves »* ou *« Je veux voir un Spider-Man »*.")

    st.divider()
    st.header("🗄️ Base de données")
    if db_ok:
        st.success(f"Connecté • **{doc_count}** films indexés")
        st.caption(f"Hôte : `{DB_HOST}:{DB_PORT}` | Base : `{DB_NAME}`")
        st.markdown(f"[🔗 Lien / Accès BDD]({DB_ADMIN_URL})")
    else:
        st.error("Connexion BDD impossible")
        st.caption(db_err)

    if st.button("🗑️ Effacer l'historique"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Que souhaitez-vous regarder aujourd'hui ?"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyse et recherche sémantique en cours..."):
            try:
                embed_response = ollama.embeddings(
                    model="nomic-embed-text",
                    prompt=user_query
                )
                query_embedding = embed_response["embedding"]

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
                            meta = row[2] if isinstance(row[2], dict) else {}
                            candidates.append({
                                "source": row[0],
                                "title": meta.get("title") or row[0],
                                "content": row[1],
                                "metadata": meta,
                                "distance": distance,
                                "similarity": similarity_pct
                            })

                valid_candidates = [c for c in candidates if c["similarity"] >= seuil_input]

                if not valid_candidates:
                    bot_reply = "❌ Je suis désolé, mais je n'ai trouvé aucun film dans mon catalogue qui corresponde à cette demande."
                    st.markdown(bot_reply)
                else:
                    candidates_text = ""
                    for idx, c in enumerate(valid_candidates, 1):
                        candidates_text += (
                            f"Candidat {idx}:\n"
                            f"- Titre : {c['title']}\n"
                            f"- Similarité : {c['similarity']:.1f}%\n"
                            f"- Détails : {c['content']}\n\n"
                        )

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
                        model="llama3.2",
                        messages=[{"role": "user", "content": prompt_llm}]
                    )
                    bot_reply = llm_response["message"]["content"]
                    st.markdown(bot_reply)

                    st.markdown("### 🎥 Films recommandés dans notre catalogue")
                    top_films = valid_candidates[:3]
                    cols = st.columns(len(top_films))

                    for i, c in enumerate(top_films):
                        meta = c["metadata"]
                        with cols[i]:
                            st.markdown(f"#### {c['title']}")
                            genre = meta.get("genre", "Film")
                            year = meta.get("year", "")
                            st.caption(f"{genre} • {year}" if year else genre)

                            poster = meta.get("poster_url")
                            if poster:
                                try:
                                    st.image(poster, use_container_width=True)
                                except Exception:
                                    st.warning("Affiche indisponible")
                                    st.markdown(f"[🖼️ Lien de l'affiche]({poster})")

                            description = meta.get("description") or c["content"]
                            if description:
                                st.write(description)

                            st.progress(min(100, max(0, int(c["similarity"]))))
                            st.caption(f"Correspondance : **{c['similarity']:.1f}%**")

                            trailer = meta.get("trailer_url")
                            if trailer:
                                st.markdown(f"[▶️ Voir la bande-annonce]({trailer})")

                            st.caption(f"🗄️ BDD (`documents`) : `{c['source']}` • [Lien BDD]({DB_ADMIN_URL})")

                with st.expander("🔍 Analyse vectorielle détaillée (Debug)"):
                    st.markdown(f"**Source BDD :** `{DB_HOST}:{DB_PORT}/{DB_NAME}` ([Lien BDD]({DB_ADMIN_URL}))")
                    for c in candidates:
                        status = "✅ Valide" if c["similarity"] >= seuil_input else "❌ Rejeté"
                        st.markdown(
                            f"**{c['title']}** (`source`: `{c['source']}`) — {status} "
                            f"(**{c['similarity']:.1f}%**, distance cosinus : `{c['distance']:.4f}`)"
                        )
                        st.progress(min(100, max(0, int(c["similarity"]))))

                st.session_state.messages.append({"role": "assistant", "content": bot_reply})

            except Exception as e:
                st.error(f"Une erreur est survenue : {str(e)}")