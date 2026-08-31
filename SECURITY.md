# Security policy

IndexSail is local and dependency-free, but TSV corpora, query arguments, and persisted indexes are untrusted inputs. The loader validates structure and allocation bounds; callers should still apply suitable file-size, storage, and process limits for their environment.

Report security-sensitive problems privately through GitHub's security advisory interface rather than a public issue. The supported version is the latest commit on the default branch.
