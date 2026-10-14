# Frontend UI guidelines

Clean, modern minimalism. Every style must serve a purpose: structure, state, hierarchy or readability. If it doesn't, remove it.

## Principles

- No decoration: no glows, gradients (except colormaps), animated illustrations, logos, eyebrow slogans or marketing copy
- Text states facts or actions only. Status: "4 of 12 matching designs · seed 42", not "Ready to explore"
- Data is the only colorful element. UI chrome stays neutral
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
- No accent hue: interactive emphasis is `ink`, data colormaps are the only saturated color
- Elevation by shadow, not border: `shadow-[var(--shadow-sm),var(--inner-highlight)]`, warm-tinted layered shadows from `style.css`
- Canvas renderers can't read CSS variables: `lib/colormap.ts` `solid` and the vtk.js background/outline in `Volume.tsx` hold the sRGB values of `--solid`, `--surface` and `--ink-faint`. Update them together
- `ink-muted` must keep ≥ 4.5:1 contrast on `paper` and `hover-strong`. Re-check with an a11y scan after changing colors

## Code layout

- `src/lib/`: app-agnostic primitives, never import from app modules
  - `ui.tsx` markup primitives, `colormap.ts` data colors, `range.ts` intervals, `format.ts` numbers and errors
  - `http.ts` `fetchOk`/`fetchJson` surface the server's `detail` as the error message
  - `useAsync.ts` is the only async-state pattern: abortable, `idle | loading | done | error`, no loose loading/error booleans
- `api.ts`: server contract types and endpoints. The server owns invariants (field `mask` = fluid with finite value), clients don't re-check them
- Components: `App` (config + dataset gate, view select, flow toolbar), `Sidebar` (sampling inputs), `Gallery` (`useFieldBatch`, design cards), `Heatmap`/`Volume` (`FieldViewProps` renderers), `Dataset` (dataset explorer view)
- State holds only what can't be derived, e.g. `Sidebar` and `Dataset` store narrowed ranges only (`withRange`), full bounds come from the server
- Views: the subtitle `Select` switches between the flow and dataset explorer (`#flow`/`#dataset` hash). Both stay mounted (`hidden`) so switching keeps state. Picked dataset rows reach the flow explorer as a `Design[]` request

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

- Shared color range across all loaded designs, shown in one compact legend pill in the toolbar with units
- Preserve physical aspect ratios. Solids in `solid`, never in a colormap color
- 3D renderer background matches `surface`

## Verification

After UI changes: `npm --prefix frontend run build`, inspect the real page in a browser, run an accessibility scan (0 violations).
