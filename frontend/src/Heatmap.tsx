import { useEffect, useRef } from 'react';
import type { FieldViewProps } from './api';
import { colorAt, solid } from './lib/colormap';

export default function Heatmap({ field, config, range, slice }: FieldViewProps) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [nx, ny] = config.dimensions;
  const aspect = ((nx - 1) * config.spacing[0]) / ((ny - 1) * config.spacing[1]);
  useEffect(() => {
    const context = canvas.current?.getContext('2d');
    if (!context) return;
    const image = context.createImageData(nx, ny);
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const index = x + nx * (y + ny * slice);
        const color = field.mask[index] ? colorAt(field.values[index], range) : solid;
        const pixel = 4 * (x + nx * (ny - 1 - y));
        image.data[pixel] = color[0];
        image.data[pixel + 1] = color[1];
        image.data[pixel + 2] = color[2];
        image.data[pixel + 3] = 255;
      }
    }
    context.putImageData(image, 0, 0);
  }, [field, nx, ny, slice, range]);
  return (
    <figure className="px-4">
      <canvas ref={canvas} width={nx} height={ny} style={{ aspectRatio: aspect }} className="block w-full rounded-lg [image-rendering:pixelated]" role="img" aria-label={`XY field slice ${slice + 1}; light gray regions are solids`} />
      <figcaption className="mt-1.5 text-right text-[11px] text-ink-muted">x · flow direction →</figcaption>
    </figure>
  );
}
