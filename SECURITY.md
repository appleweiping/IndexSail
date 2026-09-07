# Security policy

IndexSail is local and has a deliberately minimal, locked dependency graph, but TSV/TREC corpora, topics,
qrels, query arguments, and persisted indexes are untrusted inputs. The loader validates structure,
allocation bounds, compressed integer
overflow, payload checksum, and trailing data. The checksum detects accidental corruption; it does not
authenticate a file or make malicious input trusted. Callers should still apply suitable file-size, memory,
storage, and process limits for their environment.

Report security-sensitive problems privately through GitHub's security advisory interface rather than a public issue. The supported version is the latest commit on the default branch.
