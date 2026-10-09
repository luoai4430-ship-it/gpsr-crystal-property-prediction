# Data Directory

This tracked directory contains graph-construction code and small upstream atom-feature assets only. Large research objects are intentionally absent from the working copy.

For each deposited task, create a versioned snapshot with:

```text
snapshots/<dataset>/<task>/<version>/
  cif/id_prop.csv
  cif/<sample_id>.cif
  description.csv
  MatBert_embeddings.pt
  split_manifest.csv
  manifest.json
```

`manifest.json` should record source database/version or retrieval date, target definition and unit, filtering rules, split algorithm and seed, row counts, missing-text count, software versions, and SHA-256 hashes. The release should use repository-relative paths; historical absolute server paths may be retained only in a provenance field.

Do not assume the Apache-2.0 software license covers these data objects. Assign data licenses only after checking the JARVIS, Materials Project, Robocrystallographer, and pretrained-model terms.
