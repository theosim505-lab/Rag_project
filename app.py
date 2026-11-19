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


# ============================================================
# PARAMÈTRES DE LA BASE DE DONNÉES
# ============================================================

DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = int(os.getenv("POSTGRES_PORT", 5432))
DB_NAME = os.getenv("POSTGRES_DB", "rag")
DB_USER = os.getenv("POSTGRES_USER", "rag_user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

DB_ADMIN_URL = os.getenv(
    "DB_ADMIN_URL",
    f"postgresql://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

conn_string = (
    f"host={DB_HOST} "
    f"port={DB_PORT} "
    f"dbname={DB_NAME} "
    f"user={DB_USER} "
    f"password={DB_PASSWORD}"
)


def get_db_stats():
    """Vérifie la connexion PostgreSQL et compte les documents indexés."""

    try:
        with psycopg.connect(conn_string, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM documents;")
                count = cur.fetchone()[0]

        return True, count, None

    except Exception as error:
        return False, 0, str(error)


# ============================================================
# VÉRIFICATION DU SUJET
# ============================================================

def is_movie_related(user_query):
    """Détermine si la demande concerne le cinéma."""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "system",
                "content": """
Tu es le classificateur de CinéBot,
un assistant spécialisé exclusivement dans le cinéma.

Réponds YES si la question concerne :
- Les films
- Les acteurs et actrices
- Les réalisateurs
- Les genres cinématographiques
- Les sagas
- Les personnages de films
- Les recommandations de films
- L'histoire du cinéma

Réponds NO si la question concerne :
- La cuisine et les recettes
- La programmation
- Les mathématiques
- Les actualités générales
- Les conseils personnels
- Tout autre sujet sans rapport avec le cinéma.

Attention : si une question mentionne un film
mais demande une recette, réponds NO.

Réponds uniquement YES ou NO.
"""
            },
            {
                "role": "user",
                "content": user_query
            }
        ],
        options={"temperature": 0}
    )

    result = response["message"]["content"].strip().upper()

    return result == "YES"


# ============================================================
# RECHERCHE VECTORIELLE
# ============================================================

def search_movies(user_query, similarity_threshold):
    """Recherche les films les plus proches sémantiquement."""

    embed_response = ollama.embed(
        model="nomic-embed-text",
        input=user_query
    )

    query_embedding = embed_response["embeddings"][0]

    candidates = []

    with psycopg.connect(conn_string) as conn:
        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT
                    source,
                    content,
                    metadata,
                    (embedding <=> %s::vector) AS distance
                FROM documents
                ORDER BY distance ASC
                LIMIT 5;
                """,
                (query_embedding,)
            )

            rows = cur.fetchall()

    for row in rows:
        source, content, metadata, distance = row

        distance = float(distance)
        similarity = max(0.0, (1.0 - distance)) * 100

        if isinstance(metadata, dict):
            movie_metadata = metadata
        else:
            movie_metadata = {}

        candidates.append(
            {
                "source": source,
                "title": movie_metadata.get("title") or source,
                "content": content,
                "metadata": movie_metadata,
                "distance": distance,
                "similarity": similarity
            }
        )

    valid_candidates = [
        candidate
        for candidate in candidates
        if candidate["similarity"] >= similarity_threshold
    ]

    return candidates, valid_candidates


# ============================================================
# SÉLECTION DES FILMS PAR LE LLM
# ============================================================

def select_movies_with_llm(user_query, valid_candidates):
    """Le LLM choisit uniquement parmi les identifiants fournis."""

    candidates_text = ""

    for index, candidate in enumerate(valid_candidates, start=1):
        metadata = candidate["metadata"]

        candidates_text += (
            f"CANDIDAT_ID_{index}:\n"
            f"- Titre exact : {candidate['title']}\n"
            f"- Similarité : {candidate['similarity']:.1f}%\n"
            f"- Genre : {metadata.get('genre', 'Non renseigné')}\n"
            f"- Année : {metadata.get('year', 'Non renseignée')}\n"
            f"- Description : {candidate['content']}\n\n"
        )

    prompt = f"""
Tu es CinéBot, un assistant spécialisé exclusivement
dans la recommandation de films.

DEMANDE DE L'UTILISATEUR :
{user_query}

CANDIDATS DISPONIBLES DANS LE CATALOGUE :

{candidates_text}

RÈGLES STRICTES :

1. Analyse uniquement les candidats fournis.

2. Sélectionne au maximum 3 films correspondant le mieux
à la demande de l'utilisateur.

3. Réponds UNIQUEMENT avec une liste d'identifiants,
séparés par des virgules.

4. N'écris aucun texte, aucune explication, aucun titre
et aucun caractère supplémentaire.

5. Utilise uniquement les identifiants présents :
CANDIDAT_ID_1, CANDIDAT_ID_2, etc.

6. Si aucun film ne correspond, réponds uniquement :
NONE

Exemple de réponse valide :
CANDIDAT_ID_1,CANDIDAT_ID_3,CANDIDAT_ID_5

Réponds uniquement avec les identifiants ou NONE.
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={"temperature": 0}
    )

    raw_response = response["message"]["content"].strip()

    if raw_response.upper() == "NONE":
        return []

    selected_indexes = []

    for part in raw_response.split(","):
        part = part.strip().upper()

        if not part.startswith("CANDIDAT_ID_"):
            continue

        try:
            candidate_index = int(part.replace("CANDIDAT_ID_", "")) - 1
        except ValueError:
            continue

        if 0 <= candidate_index < len(valid_candidates):
            selected_indexes.append(candidate_index)

    # Supprime les doublons tout en conservant l'ordre
    selected_indexes = list(dict.fromkeys(selected_indexes))

    return [
        valid_candidates[index]
        for index in selected_indexes[:3]
    ]


# ============================================================
# GÉNÉRATION DE LA RÉPONSE FINALE
# ============================================================

def generate_final_answer(user_query, recommended_films, valid_candidates):
    """Génère une explication uniquement à partir des films retenus."""

    if not recommended_films:
        return (
            "❌ Je suis désolé, mais je n'ai trouvé aucun film "
            "dans mon catalogue qui corresponde à cette demande."
        )

    candidates_text = ""

    for candidate in valid_candidates:
        metadata = candidate["metadata"]

        candidates_text += (
            f"Titre exact : {candidate['title']}\n"
            f"Similarité : {candidate['similarity']:.1f}%\n"
            f"Genre : {metadata.get('genre', 'Non renseigné')}\n"
            f"Année : {metadata.get('year', 'Non renseignée')}\n"
            f"Description : {candidate['content']}\n\n"
        )

    recommended_titles = "\n".join(
        f"- {film['title']} — {film['similarity']:.1f}%"
        for film in recommended_films
    )

    prompt = f"""
Tu es CinéBot, un assistant spécialisé dans les recommandations de films.

DEMANDE DE L'UTILISATEUR :
{user_query}

FILMS SÉLECTIONNÉS UNIQUEMENT PARMI LE CATALOGUE :

{recommended_titles}

INFORMATIONS COMPLÈTES DES CANDIDATS :

{candidates_text}

RÈGLES STRICTES :

1. Présente uniquement les films listés dans
"FILMS SÉLECTIONNÉS UNIQUEMENT PARMI LE CATALOGUE".

2. N'invente jamais un autre film.

3. N'invente jamais un titre, une description
ou une information absente des candidats.

4. Justifie chaque film avec sa description fournie.

5. Mentionne le pourcentage de correspondance.

6. Présente au maximum 3 films.

7. Réponds en français.

FORMAT SOUHAITÉ :

Recommandations :

1. **[Titre exact]** — [pourcentage]% de correspondance  
[Justification basée uniquement sur la description fournie]

2. **[Titre exact]** — [pourcentage]% de correspondance  
[Justification basée uniquement sur la description fournie]

3. **[Titre exact]** — [pourcentage]% de correspondance  
[Justification basée uniquement sur la description fournie]
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={"temperature": 0.3}
    )

    return response["message"]["content"]


# ============================================================
# INTERFACE STREAMLIT
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .stChatInputContainer {
        padding-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🍿 CinéBot RAG")
st.markdown(
    "Votre assistant intelligent propulsé par l'IA et PostgreSQL "
    "pour trouver le film parfait."
)

db_ok, doc_count, db_error = get_db_stats()

with st.sidebar:
    st.header("⚙️ Paramètres")

    similarity_threshold = st.slider(
        "Seuil de similarité minimal (%)",
        0.0,
        100.0,
        50.0,
        5.0
    )

    st.info(
        "💡 **Astuce :** Essayez des requêtes comme "
        "*« Un film sombre sur les rêves »* ou "
        "*« Je veux voir un Spider-Man »*."
    )

    st.divider()
    st.header("🗄️ Base de données")

    if db_ok:
        st.success(f"Connecté • **{doc_count}** films indexés")
        st.caption(f"Hôte : `{DB_HOST}:{DB_PORT}` | Base : `{DB_NAME}`")
        st.markdown(f"[🔗 Lien / Accès BDD]({DB_ADMIN_URL})")
    else:
        st.error("Connexion BDD impossible")
        st.caption(db_error)

    if st.button("🗑️ Effacer l'historique"):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# HISTORIQUE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# CHAT
# ============================================================

if user_query := st.chat_input("Que souhaitez-vous regarder aujourd'hui ?"):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query
        }
    )

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):

        try:

            # --------------------------------------------------------
            # VÉRIFICATION DU SUJET
            # --------------------------------------------------------
            with st.spinner("🍿 Vérification de votre demande..."):

                if not is_movie_related(user_query):

                    bot_reply = """
🍿 **Je suis CinéBot, votre assistant spécialisé dans la recherche et la recommandation de films.**

Je peux vous aider à découvrir des films, rechercher des acteurs, des réalisateurs ou trouver un film selon vos goûts.

En revanche, je ne peux pas répondre aux demandes qui ne concernent pas le cinéma. 😊
"""

                    st.markdown(bot_reply)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": bot_reply
                        }
                    )

                    st.stop()

            # --------------------------------------------------------
            # RECHERCHE VECTORIELLE
            # --------------------------------------------------------
            with st.spinner("Analyse et recherche sémantique en cours..."):

                all_candidates, valid_candidates = search_movies(
                    user_query,
                    similarity_threshold
                )

                if not valid_candidates:

                    bot_reply = (
                        "❌ Je suis désolé, mais je n'ai trouvé aucun film "
                        "dans mon catalogue qui corresponde à cette demande."
                    )

                    st.markdown(bot_reply)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": bot_reply
                        }
                    )

                    st.stop()

                # ----------------------------------------------------
                # SÉLECTION STRICTE PAR LE LLM
                # ----------------------------------------------------
                recommended_films = select_movies_with_llm(
                    user_query,
                    valid_candidates
                )

                if not recommended_films:

                    bot_reply = (
                        "❌ Je suis désolé, mais je n'ai trouvé aucun film "
                        "dans mon catalogue qui corresponde à cette demande."
                    )

                    st.markdown(bot_reply)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": bot_reply
                        }
                    )

                    st.stop()

                # ----------------------------------------------------
                # RÉPONSE FINALE
                # ----------------------------------------------------
                bot_reply = generate_final_answer(
                    user_query,
                    recommended_films,
                    valid_candidates
                )

                st.markdown(bot_reply)

                # ----------------------------------------------------
                # AFFICHAGE DES FILMS RECOMMANDÉS
                # ----------------------------------------------------
                st.markdown("### 🎥 Films recommandés dans notre catalogue")

                top_films = recommended_films[:3]
                columns = st.columns(len(top_films))

                for index, candidate in enumerate(top_films):

                    metadata = candidate["metadata"]

                    with columns[index]:

                        st.markdown(f"#### {candidate['title']}")

                        genre = metadata.get("genre", "Film")
                        year = metadata.get("year", "")

                        if year:
                            st.caption(f"{genre} • {year}")
                        else:
                            st.caption(genre)

                        poster = metadata.get("poster_url")

                        if poster:
                            try:
                                st.image(
                                    poster,
                                    use_container_width=True
                                )
                            except Exception:
                                st.warning("Affiche indisponible")
                                st.markdown(
                                    f"[🖼️ Lien de l'affiche]({poster})"
                                )

                        description = (
                            metadata.get("description")
                            or candidate["content"]
                        )

                        if description:
                            st.write(description)

                        similarity = min(
                            100,
                            max(0, int(candidate["similarity"]))
                        )

                        st.progress(similarity)
                        st.caption(
                            f"Correspondance : "
                            f"**{candidate['similarity']:.1f}%**"
                        )

                        trailer = metadata.get("trailer_url")

                        if trailer:
                            st.markdown(
                                f"[▶️ Voir la bande-annonce]({trailer})"
                            )

                        st.caption(
                            f"🗄️ BDD (`documents`) : "
                            f"`{candidate['source']}` • "
                            f"[Lien BDD]({DB_ADMIN_URL})"
                        )

                # ----------------------------------------------------
                # DEBUG VECTORIEL
                # ----------------------------------------------------
                with st.expander("🔍 Analyse vectorielle détaillée (Debug)"):

                    st.markdown(
                        f"**Source BDD :** "
                        f"`{DB_HOST}:{DB_PORT}/{DB_NAME}` "
                        f"([Lien BDD]({DB_ADMIN_URL}))"
                    )

                    for candidate in all_candidates:

                        if candidate["similarity"] >= similarity_threshold:
                            status = "✅ Valide"
                        else:
                            status = "❌ Rejeté"

                        similarity = min(
                            100,
                            max(0, int(candidate["similarity"]))
                        )

                        st.markdown(
                            f"**{candidate['title']}** "
                            f"(`source`: `{candidate['source']}`) — "
                            f"{status} "
                            f"(**{candidate['similarity']:.1f}%**, "
                            f"distance cosinus : "
                            f"`{candidate['distance']:.4f}`)"
                        )

                        st.progress(similarity)

            # --------------------------------------------------------
            # MÉMORISATION
            # --------------------------------------------------------
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": bot_reply
                }
            )

        except Exception as error:
            st.error(f"Une erreur est survenue : {str(error)}")