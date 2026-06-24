"""Repo-map: a PageRank-ranked structural map of a TARGET repo, for navigation.

The planner gets a job spec and an (almost always empty) repo-memory digest; on a
1000-file repo that is not enough to plan against. This module builds an
aider-style **repo-map**: it extracts definition/reference tags per file with
tree-sitter, builds a def/ref graph, ranks files+symbols with PageRank, and
renders the spine (the most-referenced files and their key symbols) to a token
budget. A ``focus_files`` personalization vector collapses the map toward one
task's neighborhood, that is the per-worker SUBTREE.

This is the ONLY module the rest of grindstone imports for the feature. It is the
TYPED SEAM: every third-party call (tree-sitter, tiktoken, diskcache) is wrapped
so nothing propagates. PageRank is a small pure-Python power iteration here (the
graph is files, a few thousand nodes), so the feature pulls in no numpy/scipy/
networkx stack. Mirroring the vision gate's philosophy
(``script_vision.py``), the map ALWAYS DEGRADES, NEVER CRASHES: any error (a
missing wheel at runtime, an unparseable file, a read-only repo, an empty graph)
returns ``None`` and the run proceeds byte-identically to a run with no map.

Placement (loop / planner): the map reflects the CURRENT integration tip and
is rebuilt per planner call, so it rides the VOLATILE TAIL of the planner input,
never the byte-stable head (prefix caching depends on the head being identical
run-long). The on-disk tag cache lives under ``<repo>/.grindstone/`` (already
gitignored), never in the user's source tree.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator

__all__ = [
    "MIN_FILES_FOR_MAP",
    "PLANNER_MAP_TOKENS",
    "WORKER_SUBTREE_TOKENS",
    "build_repo_map",
    "repo_file_count",
]

#: A target repo with fewer than this many files gets NO map: codex reads a small
#: repo itself, and the greet-demo / tiny-fixture repos stay clean (and existing
#: small-repo tests see no map by design). A module-level constant, not config:
#: v1 has no flag. The threshold gates on TOTAL tracked files (``repo_file_count``).
MIN_FILES_FOR_MAP = 50

#: Whole-repo map budget injected into the planner input (the volatile tail).
PLANNER_MAP_TOKENS = 4096

#: Focused-subtree budget injected into a worker prompt (seeded on the task's
#: files); smaller than the planner's, the worker only needs its neighborhood.
WORKER_SUBTREE_TOKENS = 1024

#: Directories the tree-walker never descends: grindstone's own run state, VCS,
#: and standard build/vendor/cache trees. Pruned in-place during ``os.walk``.
_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".grindstone",
        ".git",
        "build",
        ".dart_tool",
        "node_modules",
        "__pycache__",
        ".venv",
    }
)

#: File extension -> tree-sitter grammar name, for the languages we ship a tag
#: query for. Inlined (rather than depending on grep-ast) so the supported set is
#: exactly the vendored queries, no speculative surface, and no extra untyped dep.
_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".dart": "dart",
    ".swift": "swift",
}

#: Grammar names whose tag query lives under a different vendored filename.
#: TypeScript/TSX resolve to the ``typescript`` grammar, and the JavaScript query
#: captures their function/class/method/call spine cleanly (interfaces/types are
#: out of scope for a spine map).
_QUERY_ALIASES: dict[str, str] = {"typescript": "javascript"}

#: Where the vendored ``*-tags.scm`` queries live (data only; see ATTRIBUTION.md).
_QUERY_DIR = Path(__file__).resolve().parent / "_repomap_queries"


class _Tag(NamedTuple):
    """One extracted symbol occurrence in a file (relative path)."""

    rel: str
    name: str
    kind: str  # "def" or "ref"
    line: int  # 0-based line of the occurrence
    signature: str  # stripped source line of a def (empty for refs)


# --- file walking --------------------------------------------------------------


def _iter_files(repo_root: Path) -> "Iterator[Path]":
    """Yield every regular file under ``repo_root``, pruning ``_SKIP_DIRS``."""

    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in filenames:
            yield Path(dirpath) / name


def repo_file_count(repo_root: Path) -> int:
    """Count tracked files under ``repo_root`` (respecting the skip dirs).

    The size gate for ``build_repo_map``: cheap, no third-party imports, never
    raises (an unreadable tree yields whatever it could walk, worst case 0).
    """

    try:
        return sum(1 for _ in _iter_files(repo_root))
    except OSError:
        return 0


# --- tag extraction (tree-sitter) ----------------------------------------------


def _query_path(lang: str) -> Path | None:
    """The vendored ``*-tags.scm`` for a grammar name, or ``None`` if unsupported."""

    path = _QUERY_DIR / f"{_QUERY_ALIASES.get(lang, lang)}-tags.scm"
    return path if path.is_file() else None


def _extract_tags(rel: str, source: bytes, lang: str, scm: str) -> list[_Tag]:
    """Run the language's tag query over one file's bytes; return def/ref tags.

    Local third-party imports keep ``import grindstone.repomap`` dependency-free
    (a missing wheel degrades the whole feature to None, never an import crash).
    """

    from tree_sitter import Parser, Query, QueryCursor
    from tree_sitter_language_pack import get_language

    language = get_language(lang)
    tree = Parser(language).parse(source)
    captures = QueryCursor(Query(language, scm)).captures(tree.root_node)
    lines = source.split(b"\n")
    tags: list[_Tag] = []
    for capture_name, nodes in captures.items():
        if capture_name.startswith("name.definition."):
            kind = "def"
        elif capture_name.startswith("name.reference."):
            kind = "ref"
        else:
            continue
        for node in nodes:
            name = source[node.start_byte : node.end_byte].decode("utf-8", "replace")
            row = node.start_point[0]
            signature = ""
            if kind == "def" and 0 <= row < len(lines):
                signature = lines[row].decode("utf-8", "replace").strip()
            tags.append(_Tag(rel, name, kind, row, signature))
    return tags


def _file_tags(repo_root: Path, path: Path, cache: object) -> list[_Tag]:
    """Tags for one file, memoized in ``cache`` keyed by mtime+size (or fresh)."""

    rel = path.relative_to(repo_root).as_posix()
    lang = _filename_to_lang(path.name)
    if lang is None:
        return []
    query_path = _query_path(lang)
    if query_path is None:
        return []
    try:
        stat = path.stat()
        key = f"{rel}:{stat.st_mtime_ns}:{stat.st_size}"
    except OSError:
        return []
    cached = _cache_get(cache, key)
    if cached is not None:
        return cached
    try:
        source = path.read_bytes()
        tags = _extract_tags(rel, source, lang, query_path.read_text("utf-8"))
    except Exception:  # noqa: BLE001 - any parse failure -> no tags for this file
        tags = []
    _cache_set(cache, key, tags)
    return tags


def _filename_to_lang(name: str) -> str | None:
    return _EXT_TO_LANG.get(Path(name).suffix.lower())


# --- tag cache (diskcache under <repo>/.grindstone, in-memory fallback) ---------


def _open_cache(repo_root: Path) -> object:
    """A diskcache under ``<repo>/.grindstone/repomap_cache``, else an in-mem dict.

    A read-only repo (cannot create the cache dir) falls back to a plain dict, so
    the call still works, it just re-parses every file this run.
    """

    try:
        from diskcache import Cache

        cache_dir = repo_root / ".grindstone" / "repomap_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return Cache(str(cache_dir))
    except Exception:  # noqa: BLE001 - read-only repo / missing wheel -> memory
        return {}


def _cache_get(cache: object, key: str) -> list[_Tag] | None:
    try:
        value = cache.get(key)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return None
    if isinstance(value, list):
        return [_Tag(*row) for row in value]
    return None


def _cache_set(cache: object, key: str, tags: list[_Tag]) -> None:
    try:
        cache[key] = [tuple(t) for t in tags]  # type: ignore[index]
    except Exception:  # noqa: BLE001
        pass


# --- ranking (pure-Python PageRank over the def/ref graph) ---------------------


class _RankedDef(NamedTuple):
    rel: str
    name: str
    signature: str
    rank: float


class _Edge(NamedTuple):
    dst: str
    weight: float
    ident: str


#: PageRank damping factor and power-iteration bounds (aider's defaults).
_DAMPING = 0.85
_MAX_ITERS = 100
_TOLERANCE = 1.0e-6


def _pagerank(
    nodes: set[str], out_edges: dict[str, list[_Edge]], teleport: dict[str, float]
) -> dict[str, float]:
    """Power-iteration PageRank over a weighted directed graph (no numpy/scipy).

    ``teleport`` is the (already-normalized) personalization/restart distribution;
    dangling nodes (no out-edges) redistribute their mass along it, which is also
    how ``focus_files`` collapses rank toward a neighborhood.
    """

    count = len(nodes)
    rank = {n: 1.0 / count for n in nodes}
    dangling = [n for n in nodes if not out_edges.get(n)]
    for _ in range(_MAX_ITERS):
        nxt = {n: (1.0 - _DAMPING) * teleport.get(n, 0.0) for n in nodes}
        dangling_mass = _DAMPING * sum(rank[n] for n in dangling)
        for n in nodes:
            nxt[n] += dangling_mass * teleport.get(n, 0.0)
        for src, edges in out_edges.items():
            total = sum(e.weight for e in edges)
            if total <= 0:
                continue
            share = _DAMPING * rank[src]
            for edge in edges:
                nxt[edge.dst] += share * edge.weight / total
        delta = sum(abs(nxt[n] - rank[n]) for n in nodes)
        rank = nxt
        if delta < _TOLERANCE * count:
            break
    return rank


def _rank_tags(all_tags: list[_Tag], focus_rels: set[str]) -> list[_RankedDef]:
    """Rank the def/ref graph with PageRank; return definitions spine-first.

    Each identifier defined in one file and referenced in another creates an edge
    referencer -> definer (weight = reference count). PageRank ranks files; each
    file's rank flows across its out-edges onto the definitions it references, so
    the most-referenced symbols rank highest. ``focus_rels`` seeds the restart
    distribution, teleporting only to those files collapses the ranking to their
    neighborhood (the worker subtree). Empty/unmatched focus -> uniform restart.
    """

    defines: defaultdict[str, set[str]] = defaultdict(set)
    references: defaultdict[str, Counter[str]] = defaultdict(Counter)
    signatures: dict[tuple[str, str], str] = {}
    for tag in all_tags:
        if tag.kind == "def":
            defines[tag.name].add(tag.rel)
            signatures.setdefault((tag.rel, tag.name), tag.signature)
        else:
            references[tag.name][tag.rel] += 1

    nodes: set[str] = set()
    out_edges: defaultdict[str, list[_Edge]] = defaultdict(list)
    for name, definer_rels in defines.items():
        referencers = references.get(name)
        if not referencers:
            continue
        for definer in definer_rels:
            for referencer, count in referencers.items():
                out_edges[referencer].append(_Edge(definer, float(count), name))
                nodes.add(referencer)
                nodes.add(definer)
    if not nodes:
        return []

    focus_nodes = {r for r in focus_rels if r in nodes}
    if focus_nodes:
        teleport = {n: 1.0 / len(focus_nodes) for n in focus_nodes}
    else:
        teleport = {n: 1.0 / len(nodes) for n in nodes}
    ranks = _pagerank(nodes, out_edges, teleport)

    ranked: defaultdict[tuple[str, str], float] = defaultdict(float)
    for src in nodes:
        edges = out_edges.get(src, [])
        total = sum(e.weight for e in edges)
        if total <= 0:
            continue
        src_rank = ranks.get(src, 0.0)
        for edge in edges:
            ranked[(edge.dst, edge.ident)] += src_rank * edge.weight / total

    out: list[_RankedDef] = [
        _RankedDef(rel, name, signatures.get((rel, name), ""), rank)
        for (rel, name), rank in ranked.items()
    ]
    out.sort(key=lambda d: (d.rank, d.rel, d.name), reverse=True)
    return out


# --- rendering (token-budgeted) -------------------------------------------------


def _count_tokens(text: str) -> int:
    """Token count via tiktoken, falling back to a char heuristic if unavailable.

    tiktoken's first call may fetch+cache the encoding; if that fails (offline,
    no wheel) we approximate, so token budgeting never blocks the map.
    """

    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:  # noqa: BLE001
        return max(1, len(text) // 4)


def _render(ranked: list[_RankedDef], map_tokens: int) -> str:
    """Group ranked definitions by file (file order = best rank), to a budget.

    Greedy: walk definitions spine-first, opening a ``path:`` block the first time
    a file appears and listing each definition's source line beneath it, until the
    next addition would exceed ``map_tokens``. The result surfaces the spine files
    first with their most-referenced symbols.
    """

    blocks: dict[str, list[str]] = {}
    order: list[str] = []
    used = 0
    for d in ranked:
        if d.rel not in blocks:
            header_cost = _count_tokens(f"{d.rel}:\n")
            if used + header_cost > map_tokens:
                break
            blocks[d.rel] = []
            order.append(d.rel)
            used += header_cost
        line = f"  {d.signature}" if d.signature else f"  {d.name}"
        line_cost = _count_tokens(line + "\n")
        if used + line_cost > map_tokens:
            break
        blocks[d.rel].append(line)
        used += line_cost
    rendered = "\n".join(
        f"{rel}:\n" + "\n".join(blocks[rel]) for rel in order if blocks[rel]
    )
    return rendered.strip()


# --- public entry point --------------------------------------------------------


def build_repo_map(
    repo_root: Path,
    *,
    map_tokens: int = PLANNER_MAP_TOKENS,
    focus_files: list[Path] | None = None,
) -> str | None:
    """Render a repo-map for ``repo_root``, or ``None`` (gate / any failure).

    Returns ``None`` when the repo is below ``MIN_FILES_FOR_MAP`` (skip entirely,
    behave exactly as without the feature) OR when anything at all goes wrong
    (missing wheel, parse error, read-only repo, empty graph). ``focus_files``
    seeds the PageRank personalization to collapse the map toward that task's
    neighborhood (the worker subtree); paths are matched against the repo tree, a
    non-existent or out-of-repo path is simply dropped from the seed. The tag
    cache lives under ``repo_root/.grindstone/``; never the user's source tree.
    """

    if repo_file_count(repo_root) < MIN_FILES_FOR_MAP:
        return None
    try:
        focus_rels = _resolve_focus(repo_root, focus_files)
        cache = _open_cache(repo_root)
        try:
            all_tags: list[_Tag] = []
            for path in _iter_files(repo_root):
                all_tags.extend(_file_tags(repo_root, path, cache))
        finally:
            _close_cache(cache)
        if not all_tags:
            return None
        ranked = _rank_tags(all_tags, focus_rels)
        text = _render(ranked, map_tokens)
        return text or None
    except Exception:  # noqa: BLE001 - the map is an enhancement; never crash a run
        return None


def _resolve_focus(repo_root: Path, focus_files: list[Path] | None) -> set[str]:
    """Resolve focus paths/globs to repo-relative posix strings that exist.

    Accepts concrete files, directories (all files under them), and glob patterns
    (e.g. an implement task's ``file_ownership``). Anything outside the repo or
    matching nothing on disk is dropped (a brand-new file has no graph node yet).
    """

    if not focus_files:
        return set()
    root = repo_root.resolve()
    rels: set[str] = set()
    for spec in focus_files:
        raw = str(spec)
        matches: list[Path] = []
        if any(ch in raw for ch in "*?[") and not Path(raw).is_absolute():
            matches = [p for p in root.glob(raw) if p.is_file()]
        else:
            candidate = spec if spec.is_absolute() else root / spec
            if candidate.is_dir():
                matches = [p for p in candidate.rglob("*") if p.is_file()]
            elif candidate.is_file():
                matches = [candidate]
        for match in matches:
            try:
                rels.add(match.resolve().relative_to(root).as_posix())
            except ValueError:
                continue
    return rels


def _close_cache(cache: object) -> None:
    close = getattr(cache, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001
            pass
