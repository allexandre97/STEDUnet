# Knowledge-base index

- Schema version: `1.2.0`
- Manifest: [`Papers/manifest.yaml`](../Papers/manifest.yaml)
- Manifest SHA-256: `7ad6df9430b6371501c3da52647cf1a1556903d7d6e6c63442c14bfa34dd0188`
- Latest validation date: `2026-06-18`

## Artifacts

- [`paper_cards.jsonl`](paper_cards.jsonl) — `87af2f5bd36082de2d56590024cdb4069dcc8ad574e0a4d2f1a71e3e3b93c781`
- [`synthesis_claims.jsonl`](synthesis_claims.jsonl) — `ef45ed83aae3f3b92c521f7dd2a5e85087cb93af225f8d36cafb6bdd6ae75d94`
- [`comparison_table.csv`](comparison_table.csv) — `d6fac2578454b06d7dfead01995635d3a3e4bd4040bc5261d45b53e45c084b84`
- [`synthesis.md`](synthesis.md) — `fc6017458e81a28d1024b1791d918c8c08733c4c55ad8b8242b8aed2b4c90a5e`

## Corpus status

| Cite key | Corpus role | Source PDF | Status |
|:--|:--|:--|:--|
| `Liu18` | `core` | [`Liu18.pdf`](../Papers/Liu18.pdf) | `reviewed` |
| `Shi20b` | `core` | [`Shi20b.pdf`](../Papers/Shi20b.pdf) | `reviewed` |
| `Tet18` | `core` | [`Tet18.pdf`](../Papers/Tet18.pdf) | `reviewed` |
| `Liu20` | `core` | [`Liu20.pdf`](../Papers/Liu20.pdf) | `reviewed` |
| `Liu19` | `core` | [`Liu19.pdf`](../Papers/Liu19.pdf) | `reviewed` |
| `Xu15` | `core` | [`Xu15.pdf`](../Papers/Xu15.pdf) | `reviewed` |
| `Ozd21` | `core` | [`Ozd21.pdf`](../Papers/Ozd21.pdf) | `reviewed` |
| `Togo26` | `core` | [`Togo26.pdf`](../Papers/Togo26.pdf) | `reviewed` |

## Validation

```bash
python scripts/validate_knowledge_base.py
```

Known warnings: `Togo26` is an arXiv preprint. All cards require independent human review before `verified`.
