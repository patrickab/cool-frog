import { useEffect, useRef } from 'react';
import type { FieldViewProps } from './api';
import { colorAt, solid } from './lib/colormap';

const AXIS_STEP_MM = 100;

/** XY slice at `slice`, `axis` = x ticks in mm and the flow direction below. */
export default function Heatmap({ field, config, scale, slice, axis = true }: FieldViewProps & { axis?: boolean }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [nx, ny] = config.dimensions;
  const [lengthX, lengthY] = [(nx - 1) * config.spacing[0], (ny - 1) * config.spacing[1]];
  useEffect(() => {
    const context = canvas.current?.getContext('2d');
    if (!context) return;
    const image = context.createImageData(nx, ny);
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const index = x + nx * (y + ny * slice);
        const color = field.mask[index] ? colorAt(field.values[index], scale) : solid;
        const pixel = 4 * (x + nx * (ny - 1 - y));
        image.data[pixel] = color[0];
        image.data[pixel + 1] = color[1];
        image.data[pixel + 2] = color[2];
        image.data[pixel + 3] = 255;
      }
    }
    context.putImageData(image, 0, 0);
  }, [field, nx, ny, slice, scale]);
  const lengthMm = Math.round(lengthX * 1000);
  const ticks = Array.from({ length: Math.floor(lengthMm / AXIS_STEP_MM) + 1 }, (_, index) => index * AXIS_STEP_MM);
  return (
    <figure className="px-4">
      <canvas ref={canvas} width={nx} height={ny} style={{ aspectRatio: lengthX / lengthY }} className="block w-full rounded-lg ring-1 ring-divider-soft [image-rendering:pixelated]" role="img" aria-label={`XY field slice ${slice + 1}; light gray regions are solids`} />
      {axis && <figcaption className="mt-1 text-[11px] text-ink-muted tabular-nums">
        <div aria-hidden className="relative h-3.5">
          {ticks.map((tick) => <span key={tick} className="absolute top-0 flex -translate-x-1/2 flex-col items-center leading-none first:translate-x-0 first:items-start last:-translate-x-full last:items-end" style={{ left: `${(tick / lengthMm) * 100}%` }}>
            <i className="h-1 w-px bg-ink-subtle" />{tick}
          </span>)}
        </div>
        <p className="mt-0.5 flex justify-between"><span>y 0–{Math.round(lengthY * 1000)} mm</span><span>x [mm] · flow →</span></p>
      </figcaption>}
    </figure>
  );
}
