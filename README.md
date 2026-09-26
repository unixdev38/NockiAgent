# Nocki Agent

**Un agent IA 100 % local, modulaire, sécurisé et extensible.**
LangGraph pour l'orchestration · Ollama pour l'inférence · Python moderne · SQLite · FastAPI · conçu comme un vrai projet logiciel maintenable.

> ⚠️ Le modèle LLM est **interchangeable** (`gemma3:1b`, `qwen3`, `llama3`, …). Rien n'est couplé à un modèle particulier : tout passe par la configuration. L'agent s'appelle **Nocki Agent** — le modèle Ollama n'est que son moteur.

---

## État du projet

| Phase | Contenu | Statut |
|------|---------|--------|
| **1** | Ollama + LangGraph + CLI + conversation simple | ✅ **Terminée** |
| **1.5** | Identité de l'agent (jamais « Gemma/Qwen/… », règle de langue) | ✅ **Terminée** |
| **2** | Sécurité (`app/security/`) + 8 outils + boucle reason ⇄ tools | ✅ **Terminée** |
| **3** | Mémoire persistante SQLite (sessions, court/long terme, contexte pertinent) | ✅ **Terminée** |
| **4** | Git + analyse de code → livrée en Phase 2 (`git_status/diff/log`, `search_code`, lecture seule) | ✅ **Absorbée** |
| **5** | Service partagé + API FastAPI + streaming SSE | ✅ **Terminée** |
| **6** | RAG : interfaces + chunker + retriever TF-IDF (embeddings/Chroma à venir) | ✅ **Terminée** |
| 7 | UI React | ⏳ à venir |

> Note : la Phase 4 du plan initial (Git + analyse de code) a été livrée dans la Phase 2 sous forme d'outils **strictement lecture seule** (`git_status`, `git_diff`, `git_log`, `search_code`), conformément au document de passation.

---

## Principe fondamental

Responsabilités strictement séparées :

```
LLM (inférence)  →  Ollama, derrière une interface abstraite remplaçable
Orchestration    →  LangGraph (state machine bornée : reason ⇄ tools)
Reasoning        →  nœuds du graphe
Tools            →  indépendants du LLM (validés + permission-checked par l'app)
Memory           →  SQLite (sessions + mémoire court/long terme)
RAG              →  interfaces + retriever TF-IDF (embeddings à venir)
Security         →  permissions et workspace appliqués EN DEHORS du modèle
Interface        →  CLI et API HTTP, deux couches fines sur NockiService
```

> Le LLM est considéré comme **non fiable**. Les permissions (lecture / écriture / exécution) et le bac à sable workspace sont appliqués par l'application, jamais délégués au modèle.

---

## Architecture

```
                ┌────────────────────────────┐
                │           User             │
                └─────────────┬──────────────┘
                              ▼
             ┌────────────────────────────────┐
             │  CLI (app/main.py)  ·  API (app/api/)  │   couches de présentation
             └─────────────┬──────────────────┘
                           ▼
                ┌─────────────────────┐
                │    NockiService     │   app/service.py — cœur partagé
                └─────────┬───────────┘
          ┌──────────┬────────┴────────┬─────────────┐
          ▼          ▼                 ▼             ▼
    ┌──────────┐ ┌─────────┐   ┌────────────┐ ┌──────────┐
    │ AgentRun-│ │ Tools   │   │ Memory     │ │ Security │
    │ time     │ │ (8)     │   │ (SQLite)   │ │ perms +  │
    │ LangGraph│ │ registre│   │ + RAG      │ │ sandbox  │
    └────┬─────┘ └────┬────┘   └─────┬──────┘ └──────────┘
         │            │              │
         └────────────┴──────┬───────┘
                             ▼
                     ┌───────────────┐
                     │ LLM (Ollama)  │   inférence uniquement
                     └───────────────┘
```

### Structure du code

```
NockiAgent/
├── app/
│   ├── config.py            # Configuration centralisée (Pydantic Settings + .env)
│   ├── logging_config.py    # Logs structurés (jamais de secrets)
│   ├── service.py           # NockiService — cœur partagé CLI/API
│   ├── main.py              # CLI (couche fine, aucun calcul)
│   ├── llm/
│   │   ├── base.py          # LLMProvider (invoke + tools) + exceptions
│   │   └── ollama.py        # Implémentation Ollama (bind_tools, set_model)
│   ├── agent/
│   │   ├── state.py         # État typé (TypedDict + modèles Pydantic, reducers)
│   │   ├── prompts.py       # System prompt (identité, règles de vérité, sécurité)
│   │   ├── nodes.py         # reason / tools (permission gate) / should_continue
│   │   └── graph.py         # Graphe + AgentRuntime (timeout, stream_events)
│   ├── security/
│   │   ├── validation.py    # WorkspaceValidator (anti path traversal, tailles)
│   │   ├── permissions.py   # Permission READ/WRITE/EXECUTE × ALLOW/CONFIRM/DENY
│   │   └── sandbox.py       # run_python_code (sous-processus isolé, rlimits POSIX)
│   ├── tools/
│   │   ├── base.py          # Tool / ToolResult / ToolContext / binding LangChain
│   │   ├── calculator.py    # Évaluation AST allow-list (jamais eval)
│   │   ├── filesystem.py    # read_file / list_directory
│   │   ├── search.py        # search_code (type grep, ignore deps/binaires)
│   │   ├── git.py           # git_status / git_diff / git_log (lecture seule)
│   │   ├── python_runner.py # run_python (EXECUTE, sandbox)
│   │   └── registry.py      # ToolRegistry + build_default_registry
│   ├── memory/
│   │   ├── database.py      # SQLite thread-safe (schéma, transactions)
│   │   ├── models.py        # SessionInfo / MessageRecord / MemoryItem (UTC)
│   │   ├── repository.py    # Le seul SQL (sessions, messages, mémoire)
│   │   └── manager.py       # MemoryManager (contexte pertinent, fenêtre, prune)
│   ├── rag/
│   │   ├── base.py          # Document / Chunker / EmbeddingProvider / Retriever
│   │   ├── chunking.py      # SimpleChunker (fenêtres chevauchantes)
│   │   └── keyword.py       # KeywordRetriever (TF-IDF sans dépendance)
│   └── api/
│       ├── schemas.py       # Schémas de requêtes/réponses
│       ├── server.py        # create_app() : /health /chat /chat/stream /sessions
│       └── __main__.py      # python -m app.api → uvicorn
├── tests/                   # 146 tests, sans Ollama (FakeLLM déterministe)
├── .env.example
├── pyproject.toml
└── README.md
```

---

## Outils et modèle de sécurité

| Outil | Permission | Description |
|-------|-----------|-------------|
| `calculator` | READ | Arithmétique via AST allow-list (aucun `eval`) |
| `read_file` | READ | Lecture taille-limitée, refuse binaires et évasions |
| `list_directory` | READ | Arborescence `├──` du workspace |
| `search_code` | READ | Type grep, ignore `.git/node_modules/__pycache__/…` et binaires |
| `git_status` / `git_diff` / `git_log` | READ | Inspection git **strictement lecture seule** |
| `run_python` | EXECUTE | Code Python dans un sous-processus isolé |

**Défense en profondeur :**

1. **Permissions applicatives** — le nœud `tools` vérifie `Permission` × configuration *avant* toute exécution : `DENY` et `CONFIRM` produisent un `ToolMessage` explicatif, l'action n'est pas exécutée.
2. **Re-vérification dans l'outil** — `run_python` re-teste les drapeaux d'autorisation.
3. **WorkspaceValidator** — seul point de contrôle des chemins : relatif → workspace, symlinks résolus, `PathTraversalError` sinon ; lectures plafonnées.
4. **Bac à sable best-effort** — `python -I` isolé, timeout, rlimits POSIX (CPU/AS/FSIZE). *Documenté honnêtement : ce n'est pas une barrière anti-adversaire ; la vraie sécurité vient du fait que EXECUTE est refusé par défaut.*

Configuration de sécurité (défauts sûrs) :

```env
ALLOW_SHELL=false          # exécution shell (DANGEREUX)
ALLOW_FILE_WRITE=false     # écriture de fichiers
ALLOW_PYTHON_EXEC=false    # run_python
REQUIRE_CONFIRMATION=true  # confirmation avant WRITE/EXECUTE
```

---

## Mémoire persistante (Phase 3)

- **Sessions** : chaque conversation (CLI ou API) est persistée dans SQLite (`DATABASE_PATH`), avec comptage de messages et suppression en cascade.
- **Historique** : la fenêtre (`history_window=20` derniers messages) est rechargée à chaque tour ; un `thread` de checkpointer **frais par appel** évite le double comptage.
- **Mémoire court terme** : items scopés à la session, taillés aux `short_term_keep` derniers ; **long terme** : conservés.
- **Contexte pertinent, jamais tout** : `relevant_context()` score les items (recouvrement de mots-clés × importance) et n'envoie que le top-N dans un `SystemMessage` dédié.

---

## API HTTP (Phase 5)

```bash
pip install -e ".[api]"
python -m app.api          # écoute sur API_HOST:API_PORT (défaut 0.0.0.0:8000)
```

| Route | Description |
|-------|-------------|
| `GET /health` | État (Ollama, modèle, workspace, nb outils/sessions) — jamais de 500 si Ollama est hors ligne |
| `POST /chat` | Un tour complet → `{session_id, response, tools_used, errors}` |
| `POST /chat/stream` | Le même tour en **SSE** : events `session`, `tool_call`, `tool_result`, `final`, `done` |
| `GET /sessions` | Liste des sessions |
| `GET /sessions/{id}` | Détail + messages (404 si inconnue) |
| `DELETE /sessions/{id}` | Suppression (404 si inconnue) |

Docs interactives : `http://localhost:8000/docs`.

> L'agent (bloquant) s'exécute dans un thread worker et les événements sont repontés vers la boucle d'événements via anyio — jamais de blocage de l'event loop.

---

## RAG (Phase 6)

Architecture prête **sans dépendance lourde** : `Chunker` / `EmbeddingProvider` / `Retriever` (ABC) + `SimpleChunker` (fenêtres chevauchantes) + `KeywordRetriever` (TF-IDF léger). `MemoryManager(retriever=...)` injecte les chunks retrouvés dans le bloc de contexte, à côté des mémoires pertinentes. Un futur retriever à embeddings (Ollama) + vector store (Chroma/FAISS — groupe `rag` du `pyproject.toml`) implémentera la même interface.

---

## Prérequis

- **Python 3.11+** (3.13+ recommandé).
- **[Ollama](https://ollama.com)** installé et lancé localement (optionnel pour les tests).

```bash
ollama serve
ollama pull gemma3:1b     # léger, idéal pour démarrer (ou qwen3 / llama3)
```

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                                # + outils de test (inclut la stack API)
cp .env.example .env                                   # ajuster si besoin
```

## Lancement

```bash
python -m app.main     # CLI (ou: nocki)
python -m app.api      # API (ouvrir http://localhost:8000/docs)
```

> Si Ollama est hors ligne, la CLI et l'API démarrent quand même et **échouent proprement** (erreur claire, aucun plantage).

### Commandes CLI

| Commande | Effet |
|----------|-------|
| `/help` | Aide |
| `/tools` | Outils + permissions |
| `/model [nom]` | Afficher / changer le modèle à chaud |
| `/workspace` | Workspace actif |
| `/sessions` | Sessions enregistrées |
| `/session <id>` | Reprendre une session |
| `/clear` | Nouvelle conversation |
| `/history` | Historique persistant de la session |
| `/exit` | Quitter |

---

## Tests

```bash
pip install -e ".[dev]"
pytest                  # 146 tests
ruff check app tests    # propre
```

Toute la chaîne est testée **sans serveur Ollama** grâce à un `FakeLLM` déterministe (scriptable pour piloter la boucle d'outils) : configuration, sécurité (traversal, permissions, sandbox), outils, graphe (boucle d'outils, refus EXECUTE, max iterations, événements), mémoire (fenêtre, prune, scoring, persistance, cascades), API (TestClient + SSE, service injecté) et RAG.

---

## Décisions d'architecture (ADR résumées)

1. **Interface `LLMProvider`** — l'agent ne dépend jamais d'Ollama directement ; `invoke(messages, tools)` renvoie l'`AIMessage` brut (`tool_calls` inclus).
2. **État = `TypedDict` + Pydantic + reducers** — `add_messages`/`operator.add` accumulent ; toute valeur structurée est un modèle Pydantic.
3. **Boucle bornée** — `should_continue` + `max_iterations` + `recursion_limit = 2×max+5` + timeout global + arrêt gracieux avec réponse finale.
4. **Sécurité applicative** — permissions, validateur de workspace et sandbox appliqués hors du LLM ; défauts sûrs.
5. **Une seule couche métier** — `NockiService` ; CLI et API sont des couches fines (DRY).
6. **Mémoire sélective** — seul un contexte *pertinent* (score mots-clés × importance) part vers le LLM.
7. **RAG par interfaces** — le retriever TF-IDF actuel et le futur retriever à embeddings partagent le même contrat.
8. **Pas de valeurs en dur** — modèle, URL, workspace, timeouts, limites, ports : tout vient de la config.

---

## Feuille de route

- **Phase 7** : UI React (consommer l'API SSE existante).
- Retriever à embeddings (Ollama) + Chroma/FAISS derrière l'interface `Retriever`.
- Outils WRITE (fichier, git commit) derrière `Permission.WRITE` + confirmation interactive.

## Licence

MIT.
