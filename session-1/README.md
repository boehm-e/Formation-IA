# Séance 1 – Introduction et cadrage technique Mar 28, 2025 1:00 PM
## Objectif de la séance
- Poser les bases du projet et définir le concept de RAG
- Comparer les approches (RAG vs fine-tuning) et présenter leurs avantages/inconvénients
- Identifier les types de données à traiter (PDF, vidéos, images, etc.)
- Décrire le pipeline général (ingestion → vectorisation → récupération → interrogation du LLM)
- Explorer les technologies clés : différentes bases de données vectorielles, types d'embeddings, et algorithmes de recherche
- Réaliser un exercice pratique sur l’exploration de bases de données vectorielles


## Fichiers
- chroma_db_sample.ipynb: Exemple de code pour la création d'une base de données vectorielle avec ChromaDB
- embedding_demo.ipynb: Exemple de code pour l'embedding avec Glove (king + woman - man = queen)
- word embedding representation.pdf: Utiliser [Marimo](https://marimo.io/) pour le lancer. 
    - Representation visuelle des embeddings en 2D / 3D grace a de la reduction de dimensionnalité.
    - Calcule de distance entre les mots.
    - Exemples d'analogies : king - man + woman = queen


## Comprendre le RAG et ses Alternatives
- Qu'est-ce que RAG (Retrieval Augmented Generation) ?
  - Enrichit les réponses des LLMs avec des données externes ciblées
  - Optimise la précision et limite les réponses erronées

## RAG vs Fine-tuning : Analyse Comparative
- Fine tuning : Défis majeurs incluant les risques d'inexactitude et l'aspect financier

### Analyse des Approches
#### Méthode RAG
- Points forts : traçabilité des réponses, actualisation simple des données
- Points faibles : sélection critique des données sources

#### Approche Fine-Tuning
- Points forts : spécialisation du modèle pour des usages spécifiques
- Points faibles : risques d'imprécision, complexité de déploiement


### Exercice Pratique
- Inventaire des formats de données disponibles en interne
- Analyse collective des enjeux par type :
  - documents PDF → structure et formatage
  - contenus visuels → reconnaissance de texte, analyse d'image, identification d'éléments

### Schéma du pipeline complet :

1. **Ingestion**
   - Collecte et extraction des données
   - Chunking

2. **Vectorisation**
   - Conversion des données en embeddings

3. **Recherche**
   - Utilisation d'algorithmes de recherche (HNSW, etc.)
   - Bases vectorielles
   - (Reranking?)

4. **Interrogation**
   - Intégration avec un LLM pour générer des réponses

5. **Evaluation**

### Demo Science Infuse
[https://science-infuse.beta.gouv.fr/](https://science-infuse.beta.gouv.fr/)


## Panorama des Technologies et Outils

### Bases de données vectorielles

Plusieurs solutions sont disponibles selon les besoins :

- **Pinecone** (closed source)
  - Solution payante, adaptée aux grands volumes de données
- **ChromaDB** (open source)
  - Simple d'utilisation et légère
  - Idéale pour les applications petite/moyenne échelle
- **Weaviate** (open source)
  - Base de données native vectorielle
  - Fonctionnalités de recherche avancées et personnalisables
- **pgvector**
  - Extension PostgreSQL pour la gestion de vecteurs
  - Pertinent si PostgreSQL est déjà utilisé dans le projet

### Modèles d'Embeddings

- Concept d'embedding : voir vector_demo.ipynb
- Solutions disponibles :
  - Sentence Transformers
  - Solon
  - OpenAI Embeddings
  - Modèles Hugging Face
- Critères de sélection :
  - Précision
  - Performance
  - Taille du modèle
  - Coût
  - Langues supportées

### Algorithmes de Recherche

#### Méthodes de base
- **Similarité cosinus**
  - Mesure l'angle entre deux vecteurs dans l'espace vectoriel
  - Plus l'angle est petit, plus la similarité est forte
- **Distance euclidienne**
  - Calcule la distance point à point entre vecteurs
  - Somme les distances pour un score global
- **BM25 / TF-IDF**
  - Évalue la pertinence basée sur la fréquence des termes
  - Approche non vectorielle

#### Optimisations
- **HNSW** (Hierarchical Navigable Small World)
  - Structure de graphe hiérarchique
  - Analogue à une carte avec différents niveaux de zoom
- **IVF** (Inverted File Index)
  - Clustering de l'espace vectoriel
  - Optimisation des calculs de similarité

Autres techniques : Chunking strategy, Metadatas, Graph RAG