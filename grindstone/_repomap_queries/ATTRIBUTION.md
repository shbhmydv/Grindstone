# Vendored tree-sitter tag queries

The `*-tags.scm` files in this directory are tree-sitter tag queries vendored
from the [aider](https://github.com/Aider-AI/aider) project
(`aider/queries/tree-sitter-language-pack/`), which is licensed under the
Apache License 2.0. aider in turn sources these queries from the upstream
tree-sitter grammar projects (each grammar under its own permissive license,
typically MIT). They are pure data (s-expression match patterns), carried here
so Grindstone's repo-map (`grindstone/repomap.py`) can extract definition and
reference tags per language without a network fetch.

No aider source code is vendored; only these query data files. They are
unmodified copies. The Apache-2.0 NOTICE/attribution is preserved by this file.
