# Frontend UI guidelines

Clean, modern minimalism. Every style must serve a purpose: structure, state, hierarchy or readability. If it doesn't, remove it.

## Principles

- No decoration: no glows, gradients (except colormaps), logos, eyebrow slogans or marketing copy
- Motion only to show where an element goes or what changes about it. Triggered by the user, never autoplay, auto-rotate or camera flights
- Text states facts or actions only. Status: "4 of 12 matching designs · seed 42", not "Ready to explore"
- Data is the only colorful element, UI chrome stays neutral. Exception: the tensor-axis hues in `themeColors` (`lib/colormap.ts`), never for chrome
- Show what the user controls or needs to read, nothing pinned or highlighted without a stated reason
- Prefer one control over two (dual-thumb range slider, not min and max sliders)
- Colors: role tokens from `src/style.css` only, never raw colors in components. `ink-muted` keeps ≥ 4.5:1 contrast on `paper` and `hover-strong`

## Code layout

- `src/lib/`: app-agnostic primitives, never import from app modules. `useAsync.ts` is the only async-state pattern (`idle | loading | done | error`), `http.ts` surfaces the server's `detail` as the error message
- `api.ts`: server contract types and endpoints. The server owns invariants (field `mask` = fluid with finite value), clients don't re-check them
- State holds only what can't be derived, e.g. `Sidebar` and `Dataset` store narrowed ranges only (`withRange`), full bounds come from the server
- `TensorScenes`: React Three Fiber, lazy-loaded through `Lazy3D` so three.js stays out of the first page load. Stage shapes in `STAGES` were traced with forward hooks on `tadpole.architecture.p3d`, change them only after re-tracing
- Views: the subtitle `Select` switches Cool Overview (default), Cool Flowfields, Cool Dataframe and Cool Autoencoder (`#overview`/`#flow`/`#dataset`/`#autoencoder`). All stay mounted (`hidden`) so switching keeps state
- Overview: `TOPICS` form the sidebar via `SIDEBAR_GROUPS`. Step ids are unique across topics, numbering restarts per topic
- Overview prose with numbers (`SHOWCASES` design ids, zero-shot results, fine-tuning parameter counts) was measured, changing the source means re-measuring. Toy content (convolution blob, attention scores, LoRA matrices) is labeled once as toy, never as real weights
- Tadpole teaching assumes the reader knows autoencoders. Main text carries the argument, a `Disclosure` answers one "why" or "how"

## Components

Reuse `src/lib/ui.tsx` before writing new markup: `Panel` (the only container style), `Badge`, `Disclosure`, `Layout`, `Segmented` (exclusive choices, ≤ 6 options), `Select` (never native `<select>`), `Slider`, `RangeControl`, `Colorbar`. `FieldControls` in `Gallery.tsx` drives one gallery's render mode, field, Z slice and legend.

## Typography and spacing

- Font: Outfit (bundled). Body `text-sm`, secondary `text-xs`, numbers `tabular-nums`
- Section headings: `HEADING` in `lib/ui.tsx`
- Panel padding `p-4`, gaps `gap-5`/`space-y-4`. Stay on the Tailwind scale
- Rounding: panels `rounded-2xl`, controls `rounded-xl`, inner elements `rounded-lg`

## Layout

- Sidebar (`w-72`) at the far left, sticky, holds all view inputs. Long lists go into a collapsed `Disclosure`
- Toolbar above the content: view controls and the color legend left, status text right
- Gallery: two cards per row, one below `lg`
- Dataset table fills the viewport height and scrolls endlessly, never paged. Only visible rows render, rows load in `BLOCK_SIZE` blocks from `/api/dataset/rows`, counts and histograms come from `/api/dataset/summary`

## Interaction

- Hover: color or border change only
- Hidden-until-needed controls (e.g. Reset appears once a range is narrowed)
- Loading: `animate-pulse` placeholder in the shape of the content plus one factual line
- Errors: inline `danger` text where the failure happened, no modal

## Data visualization

- Every field plot colors through one `ColorScale` (`lib/colormap.ts`): `colorAt` for canvases, `scalePosition` + `colormapStops` for vtk.js. Never map values to colors outside it
- Scale kind comes from the server's `Channel.scale`:
  - `linear`: viridis from the minimum to the 99th percentile. Never cividis, its grey midtones read as solids
  - `log` (k, ε): viridis on log10, colorbar tagged "log scale"
  - Errors: signed on `diverging`, absolute on `magnitude`, never on the field colormap
- Shared color scale across all loaded designs, one `Colorbar` per plot
- Preserve physical aspect ratios. Solids in `solid`, never a colormap color. Plot canvases get a `divider-soft` ring
- Scientific colormaps never change between themes

## Verification

After UI changes: `npm --prefix frontend run build`, inspect the real page in a browser, run an accessibility scan (0 violations).
