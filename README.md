# Nocki Agent

**Un agent IA modulaire, sécurisé et extensible — local (Ollama) ou dans le cloud (OpenRouter).**
LangGraph pour l'orchestration · Ollama ou OpenRouter pour l'inférence · serveurs **MCP** pour les outils externes · interface **web de chat** avec envoi de documents · SQLite · FastAPI.

> ⚠️ Le modèle LLM est **interchangeable** : un modèle local Ollama (`gemma3:1b`, `qwen3`, `llama3`, …) **ou** un modèle open-source hébergé sur OpenRouter (`qwen/qwen3-8b`, …). Tout passe par la configuration (`LLM_PROVIDER`). L'agent s'appelle **Nocki Agent** — le modèle n'est que son moteur.

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
| 7 | Interface web de chat (streaming, sessions, envoi de documents, MCP) | ✅ **Terminée** |
| 8 | Fournisseur cloud **OpenRouter** (modèles open-source) | ✅ **Terminée** |
| 9 | Serveurs **MCP** (outils externes, permissions dédiées) | ✅ **Terminée** |

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
│   │   ├── factory.py       # build_llm(): Ollama ou OpenRouter selon LLM_PROVIDER
│   │   ├── ollama.py        # Implémentation Ollama (bind_tools, set_model)
│   │   └── openrouter.py    # Implémentation OpenRouter (API compatible OpenAI, cloud)
│   ├── agent/
│   │   ├── state.py         # État typé (TypedDict + modèles Pydantic, reducers)
│   │   ├── prompts.py       # System prompt (identité, règles de vérité, sécurité)
│   │   ├── nodes.py         # reason / tools (permission gate) / should_continue
│   │   └── graph.py         # Graphe + AgentRuntime (timeout, stream_events)
│   ├── security/
│   │   ├── validation.py    # WorkspaceValidator (anti path traversal, tailles)
│   │   ├── permissions.py   # Permission READ/WRITE/EXECUTE × ALLOW/CONFIRM/DENY
│   │   └── sandbox.py       # run_python_code (sous-processus isolé, rlimits POSIX)
│   ├── mcp/
│   │   ├── config.py        # mcp_servers.json : validation, ${VAR}, permissions déclarées
│   │   ├── client.py        # MCPManager : sessions stdio/HTTP, boucle asyncio dédiée
│   │   └── tools.py         # MCPTool : un outil MCP exposé comme outil Nocki
│   ├── documents/
│   │   └── store.py         # Uploads sûrs dans le workspace + PDF/DOCX → texte
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
│       ├── server.py        # create_app() : API + interface web (/)
│       ├── __main__.py      # python -m app.api → uvicorn
│       └── static/          # Interface web (HTML/CSS/JS, sans étape de build)
├── tests/                   # tests sans Ollama ni clé API (FakeLLM, faux endpoint, serveur MCP de test)
├── .env.example
├── mcp_servers.example.json # modèle de configuration des serveurs MCP
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
| `<serveur>__<outil>` (MCP) | selon `permission` du serveur | Outils des serveurs MCP — `write`/`execute` soumis à `ALLOW_MCP_TOOLS` |

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

## Interface web de chat (Phase 7)

Ouvrez **http://localhost:8000** après `python -m app.api`. Vous obtenez un vrai chat :

- **Conversation en streaming** : messages, appels d'outils visibles (cliquables pour voir le résultat), réponses en Markdown.
- **Sessions** : liste des conversations, reprise, suppression (la mémoire SQLite est partagée avec la CLI).
- **Documents** : bouton ＋ / 📎 / glisser-déposer. Le fichier est déposé dans `workspace/uploads/` ; cliquez-le ensuite pour le joindre à votre message, et l'agent le lit avec ses outils (`read_file`).
  - Formats : texte, Markdown, code, CSV, JSON, etc. ; **PDF et DOCX** sont convertis en texte à l'envoi (`<fichier>.txt` à côté de l'original).
  - Limites : `UPLOAD_MAX_BYTES` (10 Mo par défaut), extensions allow-listées, nom de fichier assaini, aucune écriture hors de `uploads/`.
- **Panneau MCP** : état de chaque serveur MCP et nombre d'outils.
- Pour que l'agent puisse lire **vos propres documents**, pointez `WORKSPACE_PATH` vers le dossier qui les contient (il reste le seul accessible en lecture).

## Serveurs MCP (Phase 9)

Le [Model Context Protocol](https://modelcontextprotocol.io) permet de brancher des outils externes (système de fichiers, web, bases de données, vos propres serveurs…) sans modifier le code.

1. Copiez le modèle et adaptez-le :

   ```bash
   cp mcp_servers.example.json mcp_servers.json   # fichier ignoré par Git
   ```

2. Déclarez vos serveurs (transport **stdio** = processus local, ou **url** = HTTP streamable) :

   ```json
   {
     "mcpServers": {
       "files": {
         "command": "npx",
         "args": ["-y", "@modelcontextprotocol/server-filesystem", "./workspace"],
         "permission": "read"
       },
       "remote": {
         "url": "https://example.com/mcp",
         "headers": { "Authorization": "Bearer ${MY_REMOTE_MCP_TOKEN}" },
         "permission": "read"
       }
     }
   }
   ```

3. Redémarrez l'application. Les outils apparaissent sous le nom `<serveur>__<outil>` (ex. `files__read_file`) et sont proposés au modèle avec leur schéma d'origine.

**Sécurité des outils MCP** — ils sont considérés comme **non fiables** et passent par le même contrôle de permissions que les outils natifs :

| `permission` du serveur | Effet par défaut (`ALLOW_MCP_TOOLS=false`) |
|---|---|
| `read` | Autorisé (lecture seule déclarée) |
| `write` / `execute` (défaut) | **Refusé** tant que `ALLOW_MCP_TOOLS=false` |

Pour autoriser les outils `write`/`execute`, il faut `ALLOW_MCP_TOOLS=true` **et** `REQUIRE_CONFIRMATION=false` (sinon l'outil est signalé « confirmation requise » et n'est pas exécuté : il n'y a pas encore de confirmation interactive). Déclarez toujours le niveau le plus bas possible. Les secrets se mettent dans l'environnement (`${VAR}`), jamais dans le JSON.

Un serveur qui ne démarre pas n'empêche pas les autres : son erreur est visible dans le panneau MCP et dans `GET /mcp`.

## API HTTP (Phase 5)

```bash
pip install -e ".[api]"
python -m app.api          # écoute sur API_HOST:API_PORT (défaut 0.0.0.0:8000)
```

| Route | Description |
|-------|-------------|
| `GET /` | Interface web de chat |
| `GET /health` | État (fournisseur, modèle, workspace, nb outils/sessions/serveurs MCP) — jamais de 500 si le LLM est hors ligne |
| `GET /mcp` | Serveurs MCP : état, erreur éventuelle, outils découverts |
| `GET /files` · `POST /files` | Documents du workspace · envoi d'un document (multipart) |
| `POST /chat` | Un tour complet → `{session_id, response, tools_used, errors}` |
| `POST /chat/stream` | Le même tour en **SSE** : events `session`, `tool_call`, `tool_result`, `final`, `done` |
| `GET /sessions` | Liste des sessions |
| `GET /sessions/{id}` | Détail + messages (404 si inconnue) |
| `DELETE /sessions/{id}` | Suppression (404 si inconnue) |

Docs interactives : `http://localhost:8000/docs`.

> ⚠️ **L'API n'a pas d'authentification.** Par défaut elle écoute sur `0.0.0.0` (accessible à tout le réseau local) et permet d'envoyer des fichiers dans le workspace. Mettez `API_HOST=127.0.0.1` si vous n'en avez pas besoin depuis un autre appareil, ou placez-la derrière un reverse proxy authentifié.

> L'agent (bloquant) s'exécute dans un thread worker et les événements sont repontés vers la boucle d'événements via anyio — jamais de blocage de l'event loop.

---

## RAG (Phase 6)

Architecture prête **sans dépendance lourde** : `Chunker` / `EmbeddingProvider` / `Retriever` (ABC) + `SimpleChunker` (fenêtres chevauchantes) + `KeywordRetriever` (TF-IDF léger). `MemoryManager(retriever=...)` injecte les chunks retrouvés dans le bloc de contexte, à côté des mémoires pertinentes. Un futur retriever à embeddings (Ollama) + vector store (Chroma/FAISS — groupe `rag` du `pyproject.toml`) implémentera la même interface.

---

## Prérequis

- **Python 3.11+** (3.13+ recommandé).
- Au choix, **un fournisseur de modèle** :
  - **[Ollama](https://ollama.com)** en local (défaut) : `ollama serve` puis `ollama pull gemma3:1b`.
  - **[OpenRouter](https://openrouter.ai)** dans le cloud : une clé API (voir ci-dessous). Rien à installer localement.

## Choisir le modèle : local (Ollama) ou cloud (OpenRouter)

```env
# Local (défaut)
LLM_PROVIDER=ollama
OLLAMA_MODEL=gemma3:1b

# Cloud — modèle open-source léger, supportant les outils
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-...          # https://openrouter.ai/keys — ne jamais la committer
OPENROUTER_MODEL=qwen/qwen3-8b
```

**Pourquoi OpenRouter :** il expose une API compatible OpenAI pour des dizaines de modèles open-source (Qwen, Llama, Gemma, gpt-oss…). Seul le transport change : le graphe LangGraph, les outils, les permissions et la mémoire restent identiques.

> ⚠️ **Le choix du modèle compte.** L'agent a besoin d'un modèle qui sait appeler des outils (*tool calling*). Vérifiez la liste à jour sur <https://openrouter.ai/models?supported_parameters=tools> — les modèles gratuits tournent souvent, changez `OPENROUTER_MODEL` si l'identifiant n'existe plus. Modèles légers conseillés : `qwen/qwen3-8b`, `meta-llama/llama-3.1-8b-instruct`, `openai/gpt-oss-20b`.
>
> Si la clé est absente, l'application démarre quand même et l'erreur est affichée proprement dans l'interface.

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                                # + outils de test (inclut la stack API)
cp .env.example .env                                   # ajuster si besoin
```

## Lancement

```bash
python -m app.main     # CLI (ou: nocki)
python -m app.api      # Interface web de chat : http://localhost:8000  — API : /docs
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
pytest                  # tous les tests
ruff check app tests    # propre
```

Toute la chaîne est testée **sans serveur Ollama, sans clé OpenRouter et sans internet** : le fournisseur OpenRouter est testé avec des clients factices, les serveurs MCP avec un vrai serveur MCP stdio de démonstration (`tests/fixtures/mcp_demo_server.py`), et l'interface web via l'API. Le `FakeLLM` est déterministe (scriptable pour piloter la boucle d'outils) : configuration, sécurité (traversal, permissions, sandbox), outils, graphe (boucle d'outils, refus EXECUTE, max iterations, événements), mémoire (fenêtre, prune, scoring, persistance, cascades), API (TestClient + SSE, service injecté) et RAG.

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

- Confirmation interactive (UI) avant les outils WRITE/EXECUTE, au lieu du refus explicite actuel.
- Authentification de l'API (token) avant une exposition réseau.
- Retriever à embeddings (Ollama ou OpenRouter) + Chroma/FAISS derrière l'interface `Retriever`, pour interroger de gros volumes de documents.
- Extraction de texte pour d'autres formats (images/OCR, PPTX, XLSX).
- Outils WRITE (fichier, git commit) derrière `Permission.WRITE` + confirmation interactive.

## Licence

MIT.
