# Map schema

Closed shelves: `objects` (durable nouns), `processes` (real movements), and
`effects` (first-order change impact). Claims are `live`, `leftover`, or `ghost`
as defined in `../CONTEXT.md`.

Live runtime claims require an absolute verification date and repository-relative
`path:line` citation. Code/tests own runtime facts; `docs/memory/` owns project
state. When either changes, update routing instead of copying behavior here.
