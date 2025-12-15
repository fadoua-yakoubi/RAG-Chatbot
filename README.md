# 💬 Système RAG d'Analyse de Dialogues Téléphoniques

## 📝 Description

Implémentation complète d'un **système d'analyse de dialogues téléphoniques** utilisant l'architecture **RAG (Retrieval Augmented Generation)** avec PostgreSQL et pgvector. Ce projet permet d'analyser et d'interroger une base de dialogues de conversations réelles entre hôtesses et clients, en utilisant la recherche sémantique et l'intelligence artificielle.

Le système transforme des corpus de dialogues en une base de connaissances interrogeable, permettant d'extraire automatiquement des informations, des patterns de conversation et des bonnes pratiques du service client.

## 🎯 Objectif du Projet

L'objectif principal est de créer un **assistant intelligent capable de répondre à des questions sur les dialogues téléphoniques** :
- 📞 Comment une hôtesse accueille-t-elle un client ?
- 💡 Quelles sont les bonnes pratiques de communication ?
- 🔍 Comment sont traitées les demandes spécifiques ?
- 📊 Quels patterns de conversation émergent ?

**Cas d'usage :**
- **Formation du personnel** : Extraire les meilleures pratiques
- **Analyse qualité** : Évaluer les interactions client-hôtesse
- **Base de connaissances** : Centraliser les réponses types
- **Amélioration continue** : Identifier les points d'amélioration



## 🛠️ Technologies Utilisées

### Backend & Base de Données
| Technologie | Version | Rôle |
|------------|---------|------|
| **Python** | 3.13 | Langage principal |
| **PostgreSQL** | 16+ | Base de données relationnelle |
| **pgvector** | 0.7+ | Extension pour recherche vectorielle |
| **psycopg3** | 3.2+ | Driver PostgreSQL moderne |

### Intelligence Artificielle
| Composant | Modèle | Caractéristiques |
|-----------|--------|------------------|
| **Embeddings** | paraphrase-multilingual-MiniLM-L12-v2 | 384 dimensions, optimisé français |
| **LLM** | Llama 3.3 70B (Groq) | 70B paramètres, ultra-rapide |

### Interface & Développement
| Outil | Version | Usage |
|-------|---------|-------|
| **Streamlit** | 1.40+ | Interface web |
| **Jupyter** | Latest | Prototypage et analyse |
| **python-dotenv** | 1.0+ | Configuration sécurisée |

## 📁 Structure du Projet

```
Rag chatbot/
│
├── 📂 data/                              ← CORPUS DE DIALOGUES
│   ├── 017_00000012.txt                  Dialogue 1
│   ├── 018_00000013.txt                  Dialogue 2
│   ├── 019_00000014.txt                  Dialogue 3
│   └── ...                               (47 fichiers de dialogues)
│        └── Format : Conversations hôtesse-client annotées
│
├── 📓 notebook/
│   └── prototypage.ipynb                 ← SETUP & ANALYSE
│        ├── Installation packages
│        ├── Connexion PostgreSQL
│        ├── Création table 'dialogues'
│        ├── Chargement modèle embedding
│        ├── Lecture dialogues depuis data/
│        ├── Génération embeddings (384D)
│        ├── Insertion dans PostgreSQL
│        └── Tests de recherche sémantique
│
├── 💻 src/
│   ├── app.py                            ← APPLICATION WEB
│   │    ├── Interface Streamlit
│   │    ├── Chat interactif
│   │    ├── Recherche dans dialogues
│   │    ├── Génération réponses (Groq)
│   │    └── Affichage sources
│   │
│   └── .env                              ← CONFIGURATION
│        ├── DB_HOST, DB_NAME, DB_USER
│        ├── DB_PASSWORD
│        └── GROQ_API_KEY
│
├── 📦 requirements.txt                   ← Dépendances Python
└── 📖 README.md                          ← Documentation
```

## 🚀 Installation

### Prérequis
- **Python 3.11 ou 3.13**
- **PostgreSQL 12+** avec extension pgvector
- **Compte Groq** (gratuit) pour l'API key

### Étape 1 : Installation de l'environnement

```bash
# Créer l'environnement virtuel
python -m venv venv

# Activer (Windows)
.\venv\Scripts\activate

# Installer les dépendances
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```


### Étape 2 : Configuration PostgreSQL

1. **Installer PostgreSQL** : https://www.postgresql.org/download/
2. **Créer la base de données** :
```sql
CREATE DATABASE rag_chatbot;
```

3. **Installer pgvector** :
```bash
# Windows : Télécharger depuis
# https://github.com/pgvector/pgvector/releases
```

### Étape 3 : Obtenir une clé API Groq

1. Créer un compte sur : https://console.groq.com
2. Générer une clé API gratuite
3. Copier la clé

### Étape 4 : Configuration

Créer `src/.env` :
```env
# PostgreSQL
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rag_chatbot
DB_USER=postgres
DB_PASSWORD=votre_mot_de_passe

# Groq API (gratuit)
GROQ_API_KEY=votre_clé_groq_ici
```

## 💻 Utilisation

### ⚠️ IMPORTANT : Exécuter d'abord le notebook !

Le notebook **doit être exécuté avant l'interface Streamlit** pour :
1. ✅ Créer la table `dialogues` dans PostgreSQL
2. ✅ Charger le modèle d'embedding multilingue
3. ✅ Lire les 47 fichiers de dialogues depuis `data/`
4. ✅ Générer les embeddings (vecteurs 384D)
5. ✅ Insérer les dialogues dans la base

### Étape 1 : Prototypage avec Jupyter

```bash
jupyter notebook
```

Ouvrir `notebook/prototypage.ipynb` et **exécuter toutes les cellules** dans l'ordre.

**Résultat attendu** : ~47 dialogues insérés dans PostgreSQL

### Étape 2 : Lancer l'interface Streamlit

```bash
streamlit run src/app.py
```

Interface disponible : **http://localhost:8501**

### Exemples de questions à poser

```
✅ "Comment une hôtesse salue-t-elle un client ?"
✅ "Comment l'hôtesse répond aux demandes de stages linguistiques ?"
✅ "Quelles sont les étapes d'une conversation téléphonique ?"
✅ "Comment l'hôtesse oriente-t-elle vers d'autres organismes ?"
```

## 🔧 Architecture du Système RAG

```
┌──────────────────────────────────────────────────────┐
│         UTILISATEUR                                   │
│  "Comment saluer un client au téléphone ?"           │
└────────────────────┬─────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────┐
│       MODÈLE D'EMBEDDING MULTILINGUE                 │
│   paraphrase-multilingual-MiniLM-L12-v2             │
│                                                       │
│   Transforme la question en vecteur 384D            │
│   [0.12, -0.34, 0.56, ..., 0.23]                   │
└────────────────────┬─────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────┐
│       POSTGRESQL + pgvector                          │
│   (Base de données vectorielle - 47 dialogues)      │
│                                                       │
│   SELECT dialogue_id, contenu,                       │
│   1 - (embedding <=> query) AS similarity            │
│   FROM dialogues                                     │
│   ORDER BY similarity DESC LIMIT 3                   │
└────────────────────┬─────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────┐
│       TOP-3 DIALOGUES PERTINENTS                     │
│                                                       │
│   Dialogue 1: "<01> hôtesse: UBS bonjour..."        │
│   (similarité: 68%)                                  │
│                                                       │
│   Dialogue 2: "<02> client: oui bonjour..."         │
│   (similarité: 62%)                                  │
│                                                       │
│   Dialogue 3: "<03> hôtesse: je vous écoute..."     │
│   (similarité: 55%)                                  │
└────────────────────┬─────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────┐
│       CONTEXTE (Concaténation des 3 dialogues)       │
└────────────────────┬─────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────┐
│       GROQ API (Llama 3.3 70B)                      │
│   Génération de la réponse basée sur le contexte    │
│                                                       │
│   Prompt: "Tu es un assistant qui analyse des        │
│            dialogues. Voici le contexte : [...]      │
│            Question : [...]"                         │
└────────────────────┬─────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────┐
│       RÉPONSE GÉNÉRÉE                                │
│                                                       │
│   "D'après les dialogues, l'hôtesse salue le        │
│    client en disant 'UBS bonjour'..."               │
│                                                       │
│   + Sources : 3 dialogues avec scores de similarité │
└──────────────────────────────────────────────────────┘
```

## 📊 Format des Dialogues

Chaque fichier `.txt` dans `data/` contient un dialogue structuré :

```
<01> hotesse
     h: U B S bonjour
<02> client
     c: oui bonjour je sais pas si j'appelle au bon endroit
<03> hotesse+client
     h: je vous écoute
     c: c'est pour
<04> client
     c: c'est pour savoir si la fac pendant l'été...
...
```

**Caractéristiques :**
- Format annoté avec numéros de tours de parole
- Identification des locuteurs (hôtesse/client)
- Conversations authentiques du service UBS

## 🎓 Comment Fonctionne le RAG ?

### 1️⃣ Phase d'Indexation (Notebook)
```
Dialogues .txt → Lecture → Modèle embedding → Vecteurs 384D → PostgreSQL
```

### 2️⃣ Phase de Recherche (Chaque question)
```
Question → Embedding → Recherche similarité cosinus → Top-3 dialogues
```

### 3️⃣ Phase de Génération (Réponse)
```
Question + Contexte (dialogues) → LLM Groq → Réponse contextualisée
```

### Avantages du RAG pour l'Analyse de Dialogues

✅ **Recherche sémantique** : Trouve les dialogues pertinents même avec des formulations différentes  
✅ **Extraction automatique** : Identifie les patterns de conversation  
✅ **Réponses factuelles** : Basées sur des dialogues réels  
✅ **Traçabilité** : Affiche les sources avec scores  
✅ **Évolutif** : Ajoutez de nouveaux dialogues sans réentraînement  



## 📚 Ressources

- **Documentation Groq** : https://console.groq.com/docs
- **Documentation pgvector** : https://github.com/pgvector/pgvector
- **Sentence-Transformers** : https://www.sbert.net/
- **Streamlit** : https://docs.streamlit.io/






---

<img width="1912" height="863" alt="image" src="https://github.com/user-attachments/assets/4c7dd716-d98c-45ed-8983-f258be2f028d" />
