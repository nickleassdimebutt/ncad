# Vector texture sources

This directory holds portable vector references for refining nCAD's relief fields and menu
previews. The application never downloads texture assets at runtime.

## Pattern Monster starter set

The first curated batch comes from [Pattern Monster](https://pattern.monster/), whose free
repeatable SVG patterns are published under the MIT License. Only entries marked
`premium: "no"` were selected.

| nCAD family | Vendored source | Original |
| --- | --- | --- |
| waves | `pattern-monster/waves-3.svg` | https://pattern.monster/waves-3 |
| chevron | `pattern-monster/chevron-1.svg` | https://pattern.monster/chevron-1 |
| herringbone | `pattern-monster/herringbone-4.svg` | https://pattern.monster/herringbone-4 |
| honeycomb / hexagon | `pattern-monster/hexagon-2.svg` | https://pattern.monster/hexagon-2 |
| triangles | `pattern-monster/triangles-4.svg` | https://pattern.monster/triangles-4 |
| scales | `pattern-monster/scales-2.svg` | https://pattern.monster/scales-2 |

SVG paths and tile bounds are kept locally as the design references. Geometry remains a
continuous height field evaluated in millimetres; the surface-detail setting controls how
finely that field is tessellated into the exported STL.

These files are the visual source of truth for pattern-by-pattern refinement; nCAD does not
parse SVG paths during a build. Keeping the build field analytic preserves offline operation
and avoids turning a visual asset into a fragile runtime dependency.

See `pattern-monster/LICENSE.txt` for the vendored-source license.
