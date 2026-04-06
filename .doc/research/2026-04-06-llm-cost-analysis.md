# Analyse : coût LLM dans Graphiti et stratégies d'élimination

**Date** : 2026-04-06
**Branche** : `docs/llm-cost-analysis`
**Contexte** : graph-mem utilise Graphiti qui dépend d'un LLM pour l'ingestion. On explore comment réduire ou éliminer ce coût.

---

## 1. Comment Graphiti utilise le LLM

### Les 5 tâches LLM

| Tâche | Quand | Complexité | Modèle minimum |
|---|---|---|---|
| **Extraction d'entités** | Chaque `save_memory()` via `add_episode()` | Moyenne | 7B |
| **Extraction de relations** | Idem | Moyenne | 7B |
| **Déduplication sémantique** | Idem ("Tom" = "Thomas" ?) | **Élevée** | 7-13B |
| **Cross-encoder reranking** | Chaque `search_memory()` | Faible | 7B |
| **Résumé d'épisodes** | Optionnel | Faible | 7B |

### Pipeline d'ingestion détaillé

```
save_memory("Bruno travaille avec Julie sur Atlas en Python")
  → POST /messages (retourne 202 immédiatement)
  → AsyncWorker traite en background :
    → graphiti_core.add_episode()
      → [LLM CALL 1] Extraction d'entités : Bruno(Developer), Julie(Colleague), Atlas(Project), Python(Technology)
      → [LLM CALL 2] Extraction de relations : Bruno--works_with-->Julie, Atlas--uses-->Python, etc.
      → [LLM CALL 3?] Déduplication : "Bruno" existe déjà ? merge. "Python" existe déjà ? réutilise.
      → [EMBEDDING] Vectorisation de chaque nouvelle entité
      → Stockage dans Neo4j
```

**Coût par save** : 1-3 appels LLM + embeddings.
**Coût par search** : 1 appel LLM (reranking) + 1 embedding (query).

### Configuration actuelle

```env
# docker-compose.yml
LLM_MODEL=google/gemma-4-26b-a4b-it     # payant sur OpenRouter
EMBEDDING_MODEL=qwen/qwen3-embedding-8b   # payant sur OpenRouter
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

### Le patch ExampleLLMClient

Notre patch dans `graphiti/zep_graphiti.py` est **critique** : beaucoup de modèles open-source (Gemma, Qwen, Llama) retournent le schéma JSON au lieu des valeurs extraites. Le patch convertit les schémas Pydantic en exemples concrets via `_schema_to_example()`. Sans ce patch, l'extraction échoue systématiquement avec les modèles non-OpenAI.

---

## 2. Le modèle de données Graphiti

### 3 types de nœuds Neo4j

| Type | Rôle | Exemple |
|---|---|---|
| **EpisodicNode** | Texte brut d'entrée + timestamp | "Bruno préfère TDD" |
| **EntityNode** | Concept extrait (name, summary, embedding) | Bruno, TDD, Lyon |
| **Community** | Cluster d'entités liées | {Bruno, TDD, Lyon} |

### Les faits (EntityEdge) — la killer feature

Les relations entre entités portent des **métadonnées temporelles** :

| Champ | Signification |
|---|---|
| `created_at` | Quand le fait a été observé |
| `valid_at` | Quand le fait est devenu vrai |
| `expired_at` | Quand le fait a expiré |
| `invalid_at` | Quand le fait a été remplacé |

**Exemple de versioning temporel** :
```
Jour 1 : "On utilise MongoDB sur Atlas"
  → [Atlas] --utilise--> [MongoDB]  valid_at: jour1, invalid_at: null

Jour 30 : "On a migré vers PostgreSQL"
  → [Atlas] --utilise--> [MongoDB]  invalid_at: jour30  ← invalidé
  → [Atlas] --utilise--> [PostgreSQL]  valid_at: jour30  ← nouveau fait
```

### Recherche hybride (3 stratégies combinées)

1. **Vectorielle** — cosine similarity sur les embeddings d'entités
2. **Traversée de graphe** — suit les relations pour trouver le contexte lié
3. **Cross-encoder reranking** — LLM re-score les résultats

### Ce que Graphiti apporte vs Neo4j seul

| Fonctionnalité | Graphiti | Neo4j seul |
|---|---|---|
| Extraction d'entités depuis du texte | Auto (LLM) | Manuel (Cypher) |
| Déduplication sémantique | Auto (LLM) | Match exact seulement |
| Versioning temporel des faits | Intégré | À coder |
| Recherche hybride (vector + graphe) | Intégré | À coder |
| Communautés | Auto | Plugin GDS |
| Reranking | Intégré | Non |

---

## 3. API REST Graphiti — endpoints disponibles

| Endpoint | Méthode | LLM requis | Description |
|---|---|---|---|
| `/messages` | POST | **Oui** | Injecter du texte → extraction auto |
| `/entity-node` | POST | **Non** | Créer un nœud entité directement |
| `/entity-edge/{uuid}` | DELETE | Non | Supprimer un fait |
| `/search` | POST | **Oui** (reranking) | Recherche hybride |
| `/get-memory` | POST | **Oui** (reranking) | Recherche contextuelle |
| `/episodes/{group_id}` | GET | Non | Récupérer les épisodes |
| `/group/{group_id}` | DELETE | Non | Supprimer un groupe |
| `/clear` | POST | Non | Vider le graphe |

**Point critique** : il n'existe **pas** d'endpoint pour créer un edge/fait directement. Seul `/messages` + LLM peut créer des relations. `/entity-node` permet de créer des nœuds sans LLM, mais pas de les relier.

**`add_episode()` n'a aucun flag** pour bypasser l'extraction LLM. C'est tout ou rien.

---

## 4. Modèles gratuits OpenRouter (état avril 2026)

### Pourquoi gratuits ?

- **Subventionnés par OpenRouter** pour attirer des développeurs (freemium/upsell)
- **Partenariats providers** : Google, NVIDIA, Meta offrent du compute en échange de visibilité
- **Pas pérenne** : les modèles gratuits peuvent être retirés à tout moment

### Rate limits

| Condition | Requêtes/jour | Requêtes/minute |
|---|---|---|
| Sans crédits (compte neuf) | **~50/jour** | ~20/min |
| Avec ≥10$ de crédits | **~1000/jour** | ~20/min |

- Limites **par modèle**, pas globales
- Les requêtes échouées **comptent** dans le quota
- OpenRouter dit : *"usually not suitable for production use"*

### Modèles gratuits disponibles

| Modèle | Params | Contexte | Viable pour Graphiti |
|---|---|---|---|
| nousresearch/hermes-3-llama-3.1-405b:free | 405B | 131K | Overkill, rate limits serrés |
| nvidia/nemotron-3-super-120b-a12b:free | 120B (12B actifs) | 262K | Bon, MoE rapide |
| qwen/qwen3.6-plus:free | ? | 1M | À tester |
| qwen/qwen3-coder:free | ? | 262K | Orienté code |
| meta-llama/llama-3.3-70b-instruct:free | 70B | 65K | Solide |
| qwen/qwen3-next-80b-a3b:free | 80B (3B actifs) | 262K | MoE, rapide |
| google/gemma-3-27b-it:free | 27B | 131K | Proche de notre Gemma-4 actuel |
| openai/gpt-oss-120b:free | 120B | 131K | Nouveau, à tester |
| openai/gpt-oss-20b:free | 20B | 131K | Correct |
| nvidia/nemotron-nano-9b-v2:free | 9B | 128K | Petit mais OK |
| google/gemma-3-12b-it:free | 12B | 32K | Correct |
| nvidia/nemotron-3-nano-30b-a3b:free | 30B (3B actifs) | 256K | Léger |
| minimax/minimax-m2.5:free | ? | 196K | Inconnu |
| stepfun/step-3.5-flash:free | ? | 256K | Inconnu |
| z-ai/glm-4.5-air:free | ? | 131K | Inconnu |
| dolphin-mistral-24b-venice:free | 24B | 32K | Possible |
| google/gemma-3-4b-it:free | 4B | 32K | Trop petit |
| google/gemma-3n-e4b-it:free | ~4B | 8K | Trop petit |
| google/gemma-3n-e2b-it:free | ~2B | 8K | Trop petit |
| liquid/lfm-2.5-1.2b-*:free | 1.2B | 32K | Trop petit |
| meta-llama/llama-3.2-3b:free | 3B | 131K | Trop petit |

### Verdict sur les gratuits

**50 req/jour** sans crédits = ~3-5 sessions/jour max (chaque save = 1-3 appels LLM).
**1000 req/jour** avec 10$ de crédits = viable pour usage perso mais fragile (modèles peuvent disparaître).

---

## 5. Stratégies pour éliminer le coût LLM

### Stratégie A : Bypass LLM — Claude fait l'extraction

Ajouter un endpoint `/entity-edge` dans notre patch Graphiti pour créer des faits directement. Claude pré-extrait les entités et relations, puis les injecte sans passer par le pipeline LLM.

```
Claude extrait : {entities: [Bruno, TDD], relations: [{Bruno --préfère--> TDD}]}
  → POST /entity-node (Bruno)
  → POST /entity-node (TDD)
  → POST /entity-edge (Bruno --préfère--> TDD)   ← à créer
```

| | |
|---|---|
| **Pro** | Garde tout Graphiti (versioning, search, communautés), Claude >> Gemma pour l'extraction, coût = 0 |
| **Con** | Faut gérer la dédup et le versioning temporel nous-mêmes, contourne le cœur de Graphiti |

### Stratégie B : LLM local via Ollama

Pointer `OPENAI_BASE_URL` vers Ollama local. Zéro changement de code.

| | |
|---|---|
| **Pro** | Gratuit, privé, zéro modif code |
| **Con** | Besoin d'un GPU, latence, qualité réduite sur la dédup |

### Stratégie C : Modèle gratuit OpenRouter

Switcher `LLM_MODEL` vers un modèle gratuit (gemma-3-27b, llama-3.3-70b, etc.).

| | |
|---|---|
| **Pro** | Un changement d'env var, zéro modif code |
| **Con** | Rate limits serrés, pas pérenne, "not suitable for production" |

### Stratégie D : Claude Agent SDK comme LLM backend

Utiliser le Claude Agent SDK (qui utilise l'auth CLI/Max subscription) pour faire tourner un petit agent qui fait l'extraction. Le coût est inclus dans l'abonnement Max.

**Deux approches possibles :**

1. **Wrapper OpenAI-compatible** : créer un proxy local qui expose `/v1/chat/completions` et délègue au SDK Claude. Graphiti pointerait `OPENAI_BASE_URL` vers ce proxy. Graphiti ne sait pas qu'il parle à Claude.

2. **Hook PostToolUse** (comme claude-mem) : capturer l'activité automatiquement via un hook, et utiliser le SDK Claude (Haiku via CLI auth) pour compresser/extraire. Le modèle principal n'a jamais besoin d'appeler `save_memory` explicitement.

| | |
|---|---|
| **Pro** | Coût = 0 (Max subscription), Claude est meilleur que tout modèle OSS pour l'extraction, pas de GPU |
| **Con** | Dépend de l'abonnement Max, proxy à maintenir, latence réseau |

### Recommandation

**Court terme** : Stratégie C (modèle gratuit OpenRouter) pour tester sans effort.
**Moyen terme** : Stratégie D (Claude SDK) — la plus prometteuse car coût = 0 avec qualité maximale.
**Long terme** : Stratégie A (bypass complet) si on veut s'affranchir totalement de Graphiti comme intermédiaire LLM.

---

## 6. Questions ouvertes

- Le SDK Claude peut-il être wrappé en endpoint OpenAI-compatible facilement ?
- Quel est le rate limit réel du SDK Claude via CLI auth (Max subscription) ?
- Le patch ExampleLLMClient fonctionne-t-il avec les modèles gratuits Gemma-3 / Llama 3.3 ?
- Peut-on désactiver le cross-encoder reranking pour avoir un search 100% sans LLM ?
