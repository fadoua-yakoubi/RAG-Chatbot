"""
Application Streamlit pour le Chatbot RAG - Dialogues
Interface graphique pour interagir avec le système RAG utilisant PostgreSQL, pgvector et Groq
Basé sur des dialogues de conversations téléphoniques
"""

import streamlit as st
import psycopg
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from dotenv import load_dotenv
from pathlib import Path

# Charger les variables d'environnement
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Chatbot RAG - Dialogues",
    page_icon="💬",
    layout="wide"
)

# Initialisation du cache pour les modèles
@st.cache_resource
def load_embedding_model():
    """Charge le modèle d'embedding multilingue (en cache)"""
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_resource
def get_db_connection():
    """Crée la connexion à PostgreSQL (en cache)"""
    try:
        conn = psycopg.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            dbname=os.getenv('DB_NAME', 'rag_chatbot'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', ''),
            autocommit=True
        )
        return conn
    except Exception as e:
        st.error(f"Erreur de connexion à la base de données: {e}")
        return None

@st.cache_resource
def get_groq_client():
    """Initialise le client Groq (en cache)"""
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        st.error("GROQ_API_KEY non trouvée dans le fichier .env")
        return None
    return Groq(api_key=api_key)

def search_similar_dialogues(query, model, conn, top_k=3):
    """Recherche les dialogues similaires dans la base vectorielle"""
    try:
        # Générer l'embedding avec normalisation
        query_embedding = model.encode(query, normalize_embeddings=True)
        
        # Convertir en float Python natifs
        embedding_list = [float(x) for x in query_embedding]
        
        # Rechercher dans la base
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, dialogue_id, contenu, 1 - (embedding <=> %s::vector) AS similarity
                FROM dialogues
                ORDER BY similarity DESC
                LIMIT %s;
            """, (embedding_list, top_k))
            
            results = cur.fetchall()
        
        return results
    except Exception as e:
        st.error(f"Erreur lors de la recherche: {e}")
        return []

def generate_response(question, context, client, temperature=0.7, max_tokens=500):
    """Génère une réponse avec Groq basée sur le contexte de dialogues"""
    prompt = f"""Tu es un assistant intelligent qui répond aux questions en te basant sur des dialogues de conversations téléphoniques.

Contexte (extraits de dialogues):
{context}

Question: {question}

Réponds de manière claire et concise en français, en t'appuyant sur les informations contenues dans les dialogues. Si les dialogues ne contiennent pas d'information pertinente, dis-le clairement."""

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Erreur lors de la génération: {e}"

# Interface principale
def main():
    # Titre et description
    st.title("💬 Chatbot RAG - Analyse de Dialogues")
    st.markdown("""
    Posez vos questions et obtenez des réponses basées sur des dialogues de conversations réelles.
    Le système utilise la recherche vectorielle pour trouver les dialogues les plus pertinents.
    """)
    
    # Sidebar pour les paramètres
    with st.sidebar:
        st.header("⚙️ Paramètres")
        
        # Nombre de dialogues à récupérer
        top_k = st.slider(
            "Nombre de dialogues à rechercher",
            min_value=1,
            max_value=10,
            value=3,
            help="Plus il y a de dialogues, plus le contexte sera riche"
        )
        
        # Température pour la génération
        temperature = st.slider(
            "Température de génération",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Plus élevé = plus créatif, plus bas = plus factuel"
        )
        
        # Longueur maximale de la réponse
        max_tokens = st.slider(
            "Longueur maximale de la réponse",
            min_value=100,
            max_value=1000,
            value=500,
            step=50
        )
        
        st.divider()
        
        # Informations sur le système
        st.subheader("ℹ️ Système")
        st.markdown("""
        - **Embeddings**: paraphrase-multilingual-MiniLM-L12-v2
        - **LLM**: Llama 3.3 70B (Groq)
        - **Base**: PostgreSQL + pgvector
        - **Type**: Dialogues téléphoniques
        """)
    
    # Charger les ressources
    with st.spinner("Chargement des modèles..."):
        model = load_embedding_model()
        conn = get_db_connection()
        client = get_groq_client()
    
    if conn is None or client is None:
        st.error("Impossible d'initialiser le système. Vérifiez votre configuration.")
        return
    
    # Vérifier le nombre de dialogues dans la base
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM dialogues;")
            dialogue_count = cur.fetchone()[0]
            st.sidebar.success(f"💬 {dialogue_count} dialogues en base")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Erreur: {e}")
        st.sidebar.info("Exécutez d'abord le notebook pour charger les dialogues.")
    
    # Zone de conversation
    st.divider()
    
    # Initialiser l'historique de conversation dans la session
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Afficher l'historique des messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("💬 Dialogues sources"):
                    for i, (doc_id, dialogue_id, text, similarity) in enumerate(message["sources"], 1):
                        st.markdown(f"**Dialogue {i}** (similarité: {similarity:.2%})")
                        st.markdown(f"*ID: {dialogue_id}*")
                        st.text(text[:300] + "..." if len(text) > 300 else text)
                        st.divider()
    
    # Input utilisateur
    if question := st.chat_input("Posez votre question..."):
        # Ajouter la question à l'historique
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Afficher la question
        with st.chat_message("user"):
            st.markdown(question)
        
        # Générer la réponse
        with st.chat_message("assistant"):
            with st.spinner("🔍 Recherche dans les dialogues..."):
                # Rechercher les dialogues pertinents
                try:
                    similar_dialogues = search_similar_dialogues(question, model, conn, top_k)
                except Exception as e:
                    st.error(f"Erreur lors de la recherche: {e}")
                    similar_dialogues = []
                
                if not similar_dialogues:
                    response = "Désolé, je n'ai pas trouvé de dialogues pertinents pour répondre à votre question."
                    st.markdown(response)
                else:
                    # Créer le contexte
                    context = "\n\n".join([dialogue[2] for dialogue in similar_dialogues])
                    
                    # Générer la réponse
                    with st.spinner("🤖 Génération de la réponse..."):
                        response = generate_response(question, context, client, temperature, max_tokens)
                        st.markdown(response)
                    
                    # Afficher les sources
                    with st.expander("💬 Dialogues sources"):
                        for i, (doc_id, dialogue_id, text, similarity) in enumerate(similar_dialogues, 1):
                            st.markdown(f"**Dialogue {i}** (similarité: {similarity:.2%})")
                            st.markdown(f"*ID: {dialogue_id}*")
                            st.text(text[:300] + "..." if len(text) > 300 else text)
                            st.divider()
            
            # Ajouter la réponse à l'historique
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "sources": similar_dialogues if similar_dialogues else []
            })
    
    # Bouton pour effacer l'historique
    if st.session_state.messages:
        if st.sidebar.button("🗑️ Effacer l'historique"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()