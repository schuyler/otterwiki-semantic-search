# otterwiki-semantic-search

A plugin for [An Otter Wiki](https://otterwiki.com/) that adds vector-based
similarity search using [ChromaDB](https://www.trychroma.com/).

The existing [otterwiki-api](https://github.com/sderle/otterwiki-api) plugin
provides full-text keyword search. This plugin lets you find pages by meaning
instead of exact matches — useful for research wikis where you know what you're
looking for but not what you called it.

## Synopsis

```sh
curl -H "Authorization: Bearer $OTTERWIKI_API_KEY" \
  "http://localhost:8080/api/v1/semantic-search?q=machine+learning+approaches"
```

```json
{
  "query": "machine learning approaches",
  "results": [
    {"name": "Neural Networks", "path": "Neural Networks", "snippet": "Neural networks are...", "distance": 0.42},
    {"name": "Statistical Methods", "path": "Statistical Methods", "snippet": "Common statistical...", "distance": 0.61}
  ],
  "total": 2
}
```

Populate the index first:

```sh
curl -X POST -H "Authorization: Bearer $OTTERWIKI_API_KEY" \
  http://localhost:8080/api/v1/reindex
```

After that, the index stays current via page lifecycle hooks (if your otterwiki
version supports them) and a background sync thread that polls git for changes
every 60 seconds.

## Installation

```sh
pip install otterwiki-semantic-search
```

For in-process embedding (no external ChromaDB server):

```sh
pip install otterwiki-semantic-search[local]
```

## Configuration

All via environment variables. The only required one is `OTTERWIKI_API_KEY`
(shared with otterwiki-api if you use both).

| Variable | Default | |
|---|---|---|
| `OTTERWIKI_API_KEY` | (required) | Bearer token for API auth |
| `CHROMADB_MODE` | `server` | `server` (external ChromaDB) or `local` (in-process) |
| `CHROMADB_HOST` | `localhost` | ChromaDB host (server mode) |
| `CHROMADB_PORT` | `8000` | ChromaDB port (server mode) |
| `CHROMADB_PATH` | `/app-data/chroma` | On-disk storage path (local mode) |
| `CHROMA_COLLECTION` | `otterwiki_pages` | ChromaDB collection name |
| `CHROMA_SYNC_INTERVAL` | `60` | Seconds between background sync cycles |

## Endpoints

- `GET /api/v1/semantic-search?q=<query>&n=5` — search by similarity
- `POST /api/v1/reindex` — rebuild the entire index
- `GET /api/v1/chroma-status` — collection status and document count (no auth required)

## How it works

Pages are split into overlapping ~150-word chunks, with YAML frontmatter
stripped and stored as metadata (title, category, tags). ChromaDB embeds the
chunks and handles the vector similarity search.

The background sync thread tracks the git HEAD SHA. Each cycle, it diffs
against the last-seen commit and upserts or deletes only the changed `.md`
files. On first boot (or empty collection), it does a full reindex
automatically.

## Testing

```sh
pip install -e ".[dev]"
pytest
```

Unit tests (chunking, frontmatter) run without ChromaDB. Integration tests
use ChromaDB's `PersistentClient` with a temp directory.

## License

MIT. See LICENSE.md for details.
