# Smart Mask quality fixtures

This directory is the fixed home for Smart Mask segmentation-quality fixtures
(source image + ground-truth mask pairs used by `benchmarks/smart_mask.py`
and `ditherzam.masking.quality`). **No fixture images or ground-truth masks
are committed here yet.** This commit adds only the provenance-manifest
*format* every fixture must satisfy before it is added, per the global
constraint: do not commit generated/third-party assets until license,
provenance, checksum, size, and `.gitignore` policy are approved.

## What is (and is not) here

- `provenance.schema.example.yaml` — an annotated example of the per-fixture
  manifest format. Every value is a placeholder; it does not describe a real
  fixture and nothing in `ditherzam` or `benchmarks` parses it yet.
- No image files (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, `.webp`)
  are ever committed here. `.gitignore` blocks them at this path.

## Required fixture categories

Per the approved design spec's performance/quality program, the eventual
licensed fixture set must cover: portrait/hair, product, animal, full body,
multiple people, busy background, low contrast, transparent source, thin
structures, and no-clear-subject. Categories used by
`ditherzam.masking.quality.aggregate_quality`'s per-category Dice floor are a
subset of these: `portrait`, `product`, `animal`, `full_body`.

## Provenance policy every fixture manifest must satisfy

Each fixture (source image + ground-truth mask pair) must have its own
manifest entry recording, at minimum:

- **source URL** — exact location the source image was obtained from.
- **author** — original creator/photographer or synthesizer of the source
  image.
- **license** — the exact redistribution license/terms covering this
  specific image (not merely "royalty-free" or "assumed permissive").
- **checksum** — SHA-256 of the exact committed image bytes, so any future
  edit or corruption is detectable.
- **transformations** — any resize/crop/recompression/anonymization applied
  before committing, so the fixture is reproducible from the documented
  source.
- **ground-truth provenance** — how the ground-truth mask was produced
  (manual annotation, tool + version, annotator, review status) and its own
  license if distinct from the source image's.

A fixture with an undocumented source, an unclear or incompatible license,
or no ground-truth provenance must not be committed, regardless of how
useful it would be for testing. This mirrors the model-asset provenance
policy in
[`assets/models/smart_mask/README.md`](../../../assets/models/smart_mask/README.md).

See `provenance.schema.example.yaml` in this directory for the exact field
shape a real fixture manifest must use.
