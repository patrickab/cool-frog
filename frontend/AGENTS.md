# Frontend UI guidelines

Clean, modern minimalism. Every style must serve a purpose: structure, state, hierarchy or readability. If it doesn't, remove it.

## Principles

- No decoration: no glows, gradients (except colormaps), logos, eyebrow slogans or marketing copy
- Motion only to show where an element goes or what changes about it (tensor rearranges, shape changes in `TensorScenes`). Triggered by the user, never autoplay, auto-rotate or camera flights
- Text states facts or actions only. Status: "4 of 12 matching designs · seed 42", not "Ready to explore"
- Data is the only colorful element. UI chrome stays neutral. Exception: the five tensor-axis colors `AXIS_COLORS` (`lib/colormap.ts`), used for one axis kind each in shape formulas and 3D scenes, never for chrome
- Show what the user controls or needs to read, nothing pinned or highlighted without a stated reason
- Prefer one control over two (dual-thumb range slider, not min and max sliders)

## Theme

Color and typography follow the light mode of `~/git/gigachad-bot` (`src/frontend/app/globals.css`), without its accent colors. Tokens live in `src/style.css`. Tailwind's default palette is wiped (`--color-*: initial`), so only role tokens compile. Never use raw colors in components.

Two layers:
1. **Role tokens**: warm OKLCH (hue 85, chroma ≤ 0.02)
2. **Derived tokens**: `color-mix` of role tokens, recompute automatically

| Token | Use |
|---|---|
| `paper` | Page background, inset areas on cards (badges, inputs, expandables, loading placeholders) |
| `surface` | Cards (`Panel`), one level above `paper` |
| `surface-elevated` | Raised elements on a surface: selected segment, value pills, slider thumbs, Reset view |
| `ink` | Primary text, primary button (`bg-ink text-paper`), slider fill |
| `ink-muted` | Secondary text, labels, units |
| `ink-subtle`, `ink-faint` | Tertiary text, disabled and decorative strokes (3D outline) |
| `divider`, `divider-soft` | Borders and dividers (24% / 10% ink) |
| `hover`, `hover-strong` | Hover tints, segmented-control and slider tracks |
| `danger`, `danger-soft` | Errors only |
| `solid` | Solid/invalid voxels in plots |

- Light only. Surfaces ascend from cream toward, never to, white
- No accent hue: interactive emphasis is `ink`, data colormaps and the tensor-axis colors are the only saturated colors
- Elevation by shadow, not border: `shadow-[var(--shadow-sm),var(--inner-highlight)]`, warm-tinted layered shadows from `style.css`
- Canvas renderers can't read CSS variables: `lib/colormap.ts` `solid` and the vtk.js background/outline in `Volume.tsx` hold the sRGB values of `--solid`, `--surface` and `--ink-faint`, `SURFACE_OKLCH`/`INK_OKLCH`/`SOLID_OKLCH` the OKLCH values for three.js scenes. Update them together
- `ink-muted` must keep ≥ 4.5:1 contrast on `paper` and `hover-strong`. Re-check with an a11y scan after changing colors

## Code layout

- `src/lib/`: app-agnostic primitives, never import from app modules
  - `ui.tsx` markup primitives, `colormap.ts` data colors, `range.ts` intervals, `format.ts` numbers and errors
  - `http.ts` `fetchOk`/`fetchJson` surface the server's `detail` as the error message
  - `useAsync.ts` is the only async-state pattern: abortable, `idle | loading | done | error`, no loose loading/error booleans
- `api.ts`: server contract types and endpoints. The server owns invariants (field `mask` = fluid with finite value), clients don't re-check them
- Components: `App` (config + dataset gate, view select), `Sidebar` (sampling inputs), `Gallery` (`useFieldBatch`, `FieldControls`, design cards), `Heatmap`/`Volume` (`FieldViewProps` renderers), `Dataset` (dataframe view), `Overview` (guided tour), `TensorScenes` (three.js scenes of the Tadpole topic), `Autoencoder` (`Reconstruction` simulation/reconstruction/error grid, shared by the Zero-shot step and the Cool Autoencoder view)
- `TensorScenes`: React Three Fiber, lazy-loaded through `Lazy3D` in `Overview`, so three.js stays out of the first page load. One instanced mesh per scene (`Voxels`), positions and scales animate between layouts, `frameloop="demand"`. Stage shapes in `STAGES` were traced with forward hooks on `tadpole.architecture.p3d` (S, B, L), change them only after re-tracing
- Tadpole teaching: assume the reader knows autoencoders. Overview and Architecture explain Tadpole on its own, never Coolfrog or CoolData, which live in the CoolData and Zero-shot topics. Main text carries the argument, a `Disclosure` answers one "why" or "how" question, never a list of loose facts
- Operation scenes (`ConvolutionScene`, `AttentionScene`) are primary 3D illustrations that build up from 2D on the same persistent cubes: a `Segmented` step control goes 2D → add depth → stride 2 or windows in a grid, the selected kernel or query keeps its identity. Visually rich is welcome when it explains. Coordinate sliders and clicking cubes both select, so keyboard and touch work
- Toy content is explicit: the convolution uses a toy blob and a 3D Sobel kernel with real arithmetic in the side panel, attention uses deterministic toy cosine scores. Label them once, never as measured weights or activations
- Tensor explorers start with one sample, and the network with one crop. Previous/Next and the stage slider share state, the full layer diagram is expandable. Center network shapes with a fixed camera/world scale so grid contraction remains visible
- Scenes show Tadpole-B only, `SizeTable` compares sizes statically. Color carries meaning: viridis = field value, gray = learned features, no categorical rainbows. Every visible control names what it changes (Samples (B), Variables (C)), no toy-only knobs like crop size or downsampling factor
- Features are not physical variables, a stride footprint is not a receptive field, rearrangement preserves values but learned operations do not. Do not claim arbitrary-shift equivariance for strided convolutions or fixed windows
- State holds only what can't be derived, e.g. `Sidebar` and `Dataset` store narrowed ranges only (`withRange`), full bounds come from the server
- Views: the subtitle `Select` switches between Cool Overview (default), Cool Flowfields, Cool Dataframe and Cool Autoencoder (`#overview`/`#flow`/`#dataset`/`#autoencoder` hash). All stay mounted (`hidden`) so switching keeps state. Picked designs reach Cool Flowfields as a `Design[]` request. Only Cool Autoencoder picks a design id, the Zero-shot step always shows `TADPOLE_DESIGN`
- Overview: `TOPICS` form the sidebar via `SIDEBAR_GROUPS`. CoolData is one topic, Tadpole groups Overview, Architecture and Zero-shot autoencoding, CoolFrog holds Fine-tuning on CoolData. Zero-shot opens with how CoolData is prepared for Tadpole (scaling, solid filling, whole-domain shapes). One topic expanded at a time, the others stay mounted but `hidden`. Step ids are unique across topics, numbering restarts per topic, so "step n" in prose refers to the same topic
- Overview, CoolFrog: fine-tuning is an overview, not a solution. Recipe bullets name Coolfrog APIs and open decisions stay open, no full training code. Trainable parameter counts come from `TadpoleAutoencoder` with GIFt LoRA, the normalization numbers from the 23 cached designs and one Tadpole-B trial
- Overview, CoolData: showcase design ids in `SHOWCASES` were picked offline as nearest-neighbour pairs that differ mainly in one parameter, and their numbers in the prose were measured from the voxel fields. Changing a design id means re-measuring its prose
- Overview, Tadpole: paper facts and notation (B, C, X, Y, Z, H_X) come from `docs/literature/Tadpole.md`, U, V, W and C_l from the `tadpole` package code, N = U·V·W is ours. Zero-shot numbers were measured with `load_reconstruction` (`api/main.py`) over the 23 designs cached at the time. `/api/reconstruction` caches each design's Tadpole-B output, about 45 s per uncached design on 8 CPU cores

## Components

Reuse `src/lib/ui.tsx` before writing new markup:

- `Panel`: `rounded-2xl` `surface` with shadow and inner highlight. The only container style
- `Badge`: small count/value pill on `paper`, tabular numbers
- `Disclosure`: collapsed `details` with a rotating chevron, for long or secondary content
- `Layout`: page shell, title plus `nav` (view select) above the sticky sidebar
- `Segmented`: radio group as segmented control on a `hover-strong` track (selected = `surface-elevated` pill with `--shadow-sm`). Use for exclusive choices with ≤ 6 options
- `Select`: listbox ported from gigachad-bot `StyledSelect` (`src/frontend/components/StyledSelect.tsx`). `box` variant for form fields, `inline` for text triggers like the subtitle. Never native `<select>`, its popup ignores the theme
- `Slider`: one thumb for a value, two thumbs for a range
- `RangeControl`: name, unit, range readout and a range `Slider`, `children` (e.g. a histogram) sit above the slider
- `Colorbar`: gradient of a `ColorScale` with end and middle ticks (`formatTick`, 3 significant digits, ×10ⁿ outside [0.01, 10⁴)), `≤`/`≥` when shown values are clipped, `· log` on log scales
- `FieldControls` (in `Gallery.tsx`): render mode, field, Z slice and color legend for one gallery, state as one `FieldDisplay`

## Typography and spacing

- Font: Outfit (`@fontsource-variable/outfit`, bundled, no network fetch)
- Section headings: `text-[11px] font-medium uppercase tracking-[0.12em] text-ink-muted` (`HEADING` in `lib/ui.tsx`)
- Body `text-sm`, secondary `text-xs`. Numbers use `tabular-nums`
- Panel padding `p-4`, gaps `gap-5`/`space-y-4`. Stay on the Tailwind scale
- Rounding: panels `rounded-2xl`, controls `rounded-xl`, inner elements `rounded-lg`

## Layout

- Sidebar (`w-72`) at the far left edge, sticky, holds all view inputs
- Toolbar above the content: view controls and the color legend left, status text right
- Gallery: two cards per row, one below `lg`
- Flow sidebar: long lists (e.g. the 35 parameters) go into a collapsed `Disclosure` with a `max-h-64` scroll container
- Dataset sidebar: column filters are the view's main input, so they fill the viewport-high sidebar in their own scroll area instead of a `Disclosure`
- Dataset table: fills the viewport height and scrolls endlessly, never paged. Only visible rows render (fixed `ROW_HEIGHT`, `table-fixed`), rows load in `BLOCK_SIZE` blocks from `/api/dataset/rows`, while counts and histograms come from `/api/dataset/summary` independent of sort

## Interaction

- Hover: color or border change only, `transition`
- Hidden-until-needed controls (e.g. Reset appears once a range is narrowed)
- Loading: `animate-pulse` placeholder in the shape of the content plus one factual line
- Errors: inline `danger` text where the failure happened, no modal

## Data visualization

- Every field plot colors through one `ColorScale` (`lib/colormap.ts`): `colorAt` for canvases, `scalePosition` + `colormapStops` for vtk.js. Never map values to colors outside it
- The server's `Channel.scale` picks the scale kind, `colorScaleOf` builds it from the shown values:
  - `linear` (every field and surface channel except k, ε): viridis from the minimum to the 99th percentile, a 0 tick when the range crosses 0. One colormap for all fields so plots read the same way. Never cividis: its grey midtones read as solids
  - `log` (k, ε): viridis on log10 from the 1st percentile of positive values, solver floors clip to the bottom color, colorbar tagged "log scale" with ticks on powers of ten
  - Errors: signed on `diverging`, absolute on the single-hue `magnitude` map, never on the field colormap
- Shared color scale across all loaded designs, one `Colorbar` per plot, left-aligned above it, `≤`/`≥` for that plot's clipped values
- `Heatmap` shows x ticks in mm and the flow direction (`axis`), turn it off only when a neighbouring plot carries it
- Preserve physical aspect ratios. Solids in `solid`, never in a colormap color. Plot canvases get a `divider-soft` ring, since the error maps are near-white at 0
- 3D renderer background matches `surface`

## Verification

After UI changes: `npm --prefix frontend run build`, inspect the real page in a browser, run an accessibility scan (0 violations).
