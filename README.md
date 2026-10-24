# Pipeline RAG distribué avec Ray

> Parallélisation des deux étapes coûteuses d'un pipeline de Retrieval-Augmented Generation — le découpage des documents et le calcul des embeddings — avec Ray
> INF8810 — TP2, UQAM

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Ray](https://img.shields.io/badge/Ray-2.55-028CF0?logo=ray)
![FAISS](https://img.shields.io/badge/FAISS-IndexFlatL2-4B8BBE)
![Transformers](https://img.shields.io/badge/🤗-Transformers-FFD21E)

---

## Le problème

Dans un pipeline RAG, la recherche vectorielle et l'appel au modèle de langage sont rapides. Ce qui coûte, c'est la préparation de l'index : découper les documents en fragments, puis calculer un vecteur pour chacun. Ces deux étapes sont séquentielles dans une implémentation naïve, alors qu'elles sont **parfaitement parallélisables** — chaque document, chaque lot de fragments est indépendant des autres.

Ce projet reprend un pipeline RAG séquentiel et distribue ces deux étapes avec Ray, sans toucher au reste.

---

## Le pipeline

```
100 articles Wikipedia (Simple English)
        |
        |  Étape 1 — chunking distribué      @ray.remote process_doc
        v
   1 881 fragments de 500 caractères maximum
        |
        |  Étape 2 — embeddings distribués   @ray.remote encode_batch
        v
   matrice (1 881 x 384)                      all-MiniLM-L6-v2
        |
        v
   index FAISS IndexFlatL2 — 1 881 vecteurs
        |
        |  recherche des 5 fragments les plus proches
        v
   génération de la réponse                   google/flan-t5-base
```

---

## Ce que Ray change concrètement

| Étape | Version séquentielle | Version distribuée |
|-------|---------------------|-------------------|
| Découpage | `[process_doc(doc) for doc in ds]` | `ray.get([process_doc.remote(doc) for doc in ds])` |
| Embeddings | `model.encode(chunks)` | `ray.get([encode_batch.remote(b) for b in batches])` |
| Index, recherche, LLM | — | inchangés |

La modification tient en trois gestes : un décorateur `@ray.remote` sur la fonction, un `.remote()` à l'appel, un `ray.get()` pour rassembler les résultats. Le reste du pipeline n'a pas bougé d'une ligne.

### Étape 1 — découpage distribué

Chaque article part dans une tâche Ray indépendante. Les 100 documents sont soumis d'un coup, puis les résultats sont rassemblés et séparés en fragments et métadonnées.

```python
@ray.remote
def process_doc(doc):
    doc_chunks = chunk_text(doc['text'])
    return [(chunk, {'title': doc['title'], 'url': doc['url']}) for chunk in doc_chunks]

taches = [process_doc.remote(doc) for doc in ds]
docs_traites = ray.get(taches)
```

Résultat : **1 881 fragments** issus de 100 articles.

### Étape 2 — embeddings distribués

Les fragments sont regroupés en **38 lots de 50**, chaque lot étant encodé par une tâche Ray, puis la matrice finale est reconstituée avec `np.vstack()`.

```python
@ray.remote
def encode_batch(chunks_batch):
    # chaque worker charge son propre modèle pour éviter les conflits entre tâches Ray
    emb_model = SentenceTransformer('all-MiniLM-L6-v2')
    return emb_model.encode(chunks_batch, convert_to_numpy=True)
```

Le détail qui compte : **chaque worker instancie son propre `SentenceTransformer`**. Partager un modèle déjà chargé entre tâches Ray oblige à le sérialiser et expose à des conflits d'accès. Le recharger par worker coûte de la mémoire mais garde les tâches réellement indépendantes.

La forme de la matrice est vérifiée par assertion avant de continuer : `(1881, 384)`, soit un vecteur de 384 dimensions par fragment.

### Étape 3 — vérification de bout en bout

L'index FAISS est construit sur les embeddings produits par Ray, puis interrogé.

Requête de test : *« What happened in April 1912? »*

Les cinq fragments les plus proches sont renvoyés avec leur distance L2 et leur article source, assemblés en contexte, et transmis à `flan-t5-base` qui produit la réponse finale.

---

## Exécution

```bash
pip install -r requirements.txt
jupyter notebook rag_ray.ipynb
```

Puis **Run all**. Le carnet télécharge seul le jeu de données Wikipedia, le modèle d'embedding et le modèle de langage depuis Hugging Face.

> Exécuté sur Google Colab avec Ray 2.55.1 et Python 3.12. `ray.init()` crée un cluster local qui exploite les CPU disponibles ; aucune configuration de cluster n'est nécessaire.

---

## Composants

| Rôle | Choix | Pourquoi |
|---|---|---|
| Jeu de données | `wikimedia/wikipedia`, `20231101.simple`, 100 articles | Texte propre et court, idéal pour valider un pipeline |
| Découpage | Par paragraphe, 500 caractères maximum | Respecte les frontières naturelles du texte avant de couper par mots |
| Embeddings | `all-MiniLM-L6-v2` | 384 dimensions, rapide sur CPU |
| Index vectoriel | FAISS `IndexFlatL2` | Recherche exacte, suffisante à cette échelle |
| Modèle de langage | `google/flan-t5-base` | Modèle seq2seq léger, exécutable sans GPU |
| Parallélisation | Ray 2.55 | Décorateur sur fonction, sans réécriture du pipeline |

---

## Équipe

Projet réalisé en équipe de trois dans le cadre du cours INF8810 à l'UQAM.

- Mamadou Rafiou Barry
- Fatima Barry
- Naouel Ouazar

**Cours :** INF8810 — Systèmes distribués et ingénierie de données pour l'apprentissage automatique
**Établissement :** UQAM — Université du Québec à Montréal
