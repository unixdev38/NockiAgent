# Nocki Agent

**Un agent IA 100 % local, modulaire, sécurisé et extensible.**
LangGraph pour l'orchestration · Ollama pour l'inférence · Python moderne · SQLite (à venir) · conçu comme un vrai projet logiciel maintenable.

> ⚠️ Le modèle LLM est **interchangeable** (`gemma3:1b`, `qwen3`, `llama3`, …). Rien n'est couplé à un modèle particulier : tout passe par la configuration.

---

## État du projet : Phase 1 ✅

| Phase | Contenu | Statut |
|------|---------|--------|
| **1** | Ollama + LangGraph + CLI + conversation simple | ✅ **Terminée** |
| 2 | Tools : calculator, list_directory, read_file, search_code, git (lecture), python_runner (sandbox) | ⏳ à venir |
| 3 | Mémoire persistante SQLite (conversation / court terme / long terme) | ⏳ |
| 4 | Git + analyse de code | ⏳ |
| 5 | API FastAPI + streaming (SSE/WebSocket) | ⏳ |
| 6 | RAG + embeddings + vector store | ⏳ |
| 7 | UI React | ⏳ |

L'architecture de la Phase 1 pose déjà les fondations (état étendu, interface LLM
remplaçable, permissions en config, structure de dossiers complète) pour accueillir
les phases suivantes **sans refonte**.

---

## Principe fondamental

Responsabilités strictement séparées :

```
LLM (inférence)  →  Ollama, derrière une interface abstraite remplaçable
Orchestration    →  LangGraph (state machine bornée)
Reasoning        →  nœuds du graphe
Tools            →  indépendants du LLM (Phase 2)
Memory           →  SQLite (Phase 3)
Security         →  permissions appliquées EN DEHORS du modèle
Interface        →  CLI (Phase 1), API (Phase 5), UI (Phase 7)
```

> Le LLM est considéré comme **non fiable**. Les permissions et restrictions
> (workspace, écriture, shell) sont appliquées par l'application, jamais déléguées
> au modèle.

---

## Architecture

```
                    ┌─────────────────┐
                    │      User       │
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐   CLI (Phase 1) / API (Phase 5)
                    │   Interface     │
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │  AgentRuntime   │   timeout · session · erreurs
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │    LangGraph    │   State · reason · should_continue
                    └────────┬────────┘
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
     ┌─────────┐        ┌─────────┐       ┌─────────┐
     │  Tools  │        │ Memory  │       │  LLM    │
     │(Phase 2)│        │(Phase 3)│       │ Ollama  │
     └─────────┘        └─────────┘       └─────────┘
```

### Structure du code

```
NockiAgent/
├── app/
│   ├── config.py            # Configuration centralisée (Pydantic Settings + .env)
│   ├── logging_config.py    # Logs structurés (jamais de secrets)
│   ├── main.py              # CLI (I/O uniquement, pas de logique métier)
│   ├── llm/
│   │   ├── base.py          # Interface LLMProvider (remplaçable) + exceptions
│   │   └── ollama.py        # Implémentation Ollama (+ checks disponibilité/modèle)
│   └── agent/
│       ├── state.py         # État typé (TypedDict + modèles Pydantic)
│       ├── prompts.py       # System prompt (règles de vérité + sécurité)
│       ├── nodes.py         # Nœuds du graphe + routeur should_continue
│       └── graph.py         # Assemblage LangGraph + AgentRuntime (timeout/stream)
├── tests/                   # Tests unitaires (config, state, graph, llm) + FakeLLM
├── .env.example
├── pyproject.toml
└── README.md
```

---

## Prérequis

- **Python 3.11+** (3.13+ recommandé).
- **[Ollama](https://ollama.com)** installé et lancé localement.

```bash
# 1) Démarrer le serveur Ollama (dans un terminal dédié)
ollama serve

# 2) Télécharger un modèle (interchangeable)
ollama pull gemma3:1b     # léger, idéal pour démarrer
# ou : ollama pull qwen3 / ollama pull llama3
```

---

## Installation

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Dépendances (mode éditable)
pip install -e .
# + outils de dev/tests
pip install -e ".[dev]"
```

### Configuration

```bash
# Copier l'exemple puis ajuster si besoin
cp .env.example .env        # Windows: copy .env.example .env
```

Variables clés (voir `.env.example` pour la liste complète) :

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:1b
MAX_ITERATIONS=8
AGENT_TIMEOUT=120
WORKSPACE_PATH=./workspace       # ex. Windows : C:/Users/HP/Projects
ALLOW_SHELL=false
ALLOW_FILE_WRITE=false
REQUIRE_CONFIRMATION=true
```

---

## Lancement (CLI)

```bash
python -m app.main
# ou, via le script installé :
nocki
```

```
╭──────────────────────────────╮
│        Nocki Agent           │
│        Local AI Agent        │
╰──────────────────────────────╯
Model:     gemma3:1b
Workspace: .../workspace
You:
>
```

### Commandes

| Commande | Effet |
|----------|-------|
| `/help` | Aide |
| `/tools` | Lister les outils (aucun en Phase 1) |
| `/model [nom]` | Afficher / changer de modèle à chaud |
| `/workspace` | Afficher le workspace |
| `/clear` | Nouvelle conversation |
| `/history` | Historique de la session |
| `/exit` | Quitter |

> Si Ollama est hors ligne, la CLI démarre quand même et affiche une erreur
> **propre** (aucun plantage) — conformément à la gestion d'erreurs voulue.

---

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Les tests utilisent un **FakeLLM** déterministe : toute la chaîne (config, état,
graphe, streaming, gestion « Ollama hors ligne ») est testée **sans** serveur Ollama.

Cas couverts en Phase 1 : configuration & overrides `.env`, état typé, agent sans
outil, LLM indisponible, arrêt sur `max_iterations`, timeout, détection de modèle.

---

## Décisions d'architecture (ADR résumées)

1. **Interface `LLMProvider`** — l'agent ne dépend jamais d'Ollama directement.
   Remplacer le backend = ajouter un module dans `app/llm/`.
2. **État = `TypedDict` + Pydantic** — le conteneur de canaux LangGraph est un
   `TypedDict` (exigé par les reducers) ; toute valeur structurée est un modèle
   Pydantic (`ToolCall`, `Observation`, `Plan`). Zéro dict non structuré qui traîne.
3. **Boucle bornée** — `should_continue` + `max_iterations` + `recursion_limit` +
   timeout global : impossible de boucler à l'infini.
4. **Sécurité en configuration** — `ALLOW_SHELL`, `ALLOW_FILE_WRITE`,
   `REQUIRE_CONFIRMATION` sont posés dès la Phase 1 et appliqués par l'app, pas par le LLM.
5. **Pas de valeurs en dur** — modèle, URL, workspace, timeouts : tout vient de la config.

---

## Feuille de route

Voir le tableau des phases ci-dessus. La prochaine étape (**Phase 2**) ajoute les
outils dans `app/tools/` et le module `app/security/`, branchés au graphe via un
nœud *tool router* — sans modifier le cœur de l'agent.

## Licence

MIT.
