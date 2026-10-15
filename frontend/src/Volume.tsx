import { useCallback, useEffect, useRef, useState } from 'react';
import type { RefObject } from 'react';
import '@kitware/vtk.js/Rendering/Profiles/Geometry';
import '@kitware/vtk.js/Rendering/Profiles/Volume';
import vtkGenericRenderWindow from '@kitware/vtk.js/Rendering/Misc/GenericRenderWindow';
import type { vtkGenericRenderWindow as GenericRenderWindow } from '@kitware/vtk.js/Rendering/Misc/GenericRenderWindow';
import type { vtkRenderer } from '@kitware/vtk.js/Rendering/Core/Renderer';
import vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData';
import vtkPolyData from '@kitware/vtk.js/Common/DataModel/PolyData';
import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray';
import vtkImageMapper from '@kitware/vtk.js/Rendering/Core/ImageMapper';
import { SlicingMode } from '@kitware/vtk.js/Rendering/Core/ImageMapper/Constants';
import vtkImageSlice from '@kitware/vtk.js/Rendering/Core/ImageSlice';
import vtkColorTransferFunction from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction';
import vtkCubeSource from '@kitware/vtk.js/Filters/Sources/CubeSource';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import { colormapStops, scalePosition, solid } from './lib/colormap';
import type { ColorScale } from './lib/colormap';
import { errorMessage } from './lib/format';
import type { Config, FieldViewProps, Surface as SurfaceData } from './api';

type Own = <T extends { delete: () => void }>(object: T) => T;
/** Adds actors to `renderer`, returns what the view updates without rebuilding. */
type Build<Controls> = (renderer: vtkRenderer, own: Own) => Controls;
/** Scalar of masked voxels, below the [0, 1] color positions so it gets the solid color. */
const MASKED_POSITION = -0.02;

/** Colormap of `scale` over color positions [0, 1], see `scalePosition`, masked and NaN values get the solid color. */
function lookupTable(scale: ColorScale, own: Own) {
  const lookup = own(vtkColorTransferFunction.newInstance());
  lookup.addRGBPoint(MASKED_POSITION, solid[0] / 255, solid[1] / 255, solid[2] / 255);
  lookup.set({ nanColor: [solid[0] / 255, solid[1] / 255, solid[2] / 255, 1] });
  const stops = colormapStops(scale.colormap);
  stops.forEach((color, index) => lookup.addRGBPoint(index / (stops.length - 1), color[0] / 255, color[1] / 255, color[2] / 255));
  return lookup;
}

/** vtk.js window with the channel outline, rebuilt when `build` or `config` change, camera oblique over the channel. */
function useScene<Controls>(config: Config, build: Build<Controls>) {
  const container = useRef<HTMLDivElement>(null);
  const scene = useRef<{ controls: Controls; reset: () => void } | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    const host = container.current;
    if (!host) return;
    setError('');
    const resources: { delete: () => void }[] = [];
    const own: Own = (object) => { resources.push(object); return object; };
    let genericWindow: GenericRenderWindow | undefined;
    let disposed = false;
    let observer: ResizeObserver | undefined;
    const dispose = () => {
      if (disposed) return;
      disposed = true;
      scene.current = null;
      observer?.disconnect();
      genericWindow?.delete();
      for (const object of resources.reverse()) object.delete();
      host.replaceChildren();
    };
    try {
      // `--surface` from style.css, so the canvas blends into its card
      genericWindow = vtkGenericRenderWindow.newInstance({ background: [0.98, 0.965, 0.933], listenWindowResize: false });
      genericWindow.setContainer(host);
      const renderWindow = own(genericWindow.getRenderWindow());
      const renderer = own(genericWindow.getRenderer());
      own(genericWindow.getInteractor());
      const controls = build(renderer, own);
      const lengths = config.dimensions.map((size, axis) => (size - 1) * config.spacing[axis]);
      const center: [number, number, number] = [lengths[0] / 2, lengths[1] / 2, lengths[2] / 2];
      const box = own(vtkCubeSource.newInstance({ xLength: lengths[0], yLength: lengths[1], zLength: lengths[2], center }));
      const mapper = own(vtkMapper.newInstance());
      mapper.setInputConnection(box.getOutputPort());
      const outline = own(vtkActor.newInstance());
      outline.setMapper(mapper);
      outline.getProperty().setRepresentationToWireframe();
      outline.getProperty().setColor(0.655, 0.643, 0.624);
      outline.getProperty().setLighting(false);
      renderer.addActor(outline);
      const reset = () => {
        const camera = renderer.getActiveCamera();
        camera.setFocalPoint(...center);
        camera.setPosition(center[0] + lengths[0] * 0.2, center[1] - lengths[0] * 0.85, center[2] + lengths[0] * 0.8);
        camera.setViewUp(0, 0, 1);
        renderer.resetCamera();
        camera.zoom(1.5);
        renderer.resetCameraClippingRange();
        renderWindow.render();
      };
      scene.current = { controls, reset };
      observer = new ResizeObserver(() => genericWindow?.resize());
      observer.observe(host);
      genericWindow.resize();
      reset();
    } catch (failure) {
      setError(`3D rendering unavailable: ${errorMessage(failure)}. Try the 2D view.`);
      dispose();
    }
    return dispose;
  }, [build, config]);

  return { container, scene, error };
}

function SceneView({ container, error, label, onReset }: { container: RefObject<HTMLDivElement | null>; error: string; label: string; onReset: () => void }) {
  return (
    <div className="relative mx-4 h-72 overflow-hidden rounded-lg">
      <div className="absolute inset-0 [&_canvas]:block [&_canvas]:size-full" ref={container} role="img" aria-label={label} />
      {error ? <p className="absolute inset-0 bg-surface p-4 text-xs text-danger" role="alert">{error}</p> : <button className="absolute right-2 bottom-2 rounded-lg border border-divider-soft bg-surface-elevated/90 px-2.5 py-1 text-[11px] text-ink-muted shadow-[var(--shadow-sm)] backdrop-blur transition hover:text-ink" onClick={onReset}>Reset view</button>}
    </div>
  );
}

/** Three orthogonal voxel slices, the Z slice follows `slice`. */
export default function Volume({ field, config, scale, slice }: FieldViewProps) {
  const currentSlice = useRef(slice);
  currentSlice.current = slice;
  const build = useCallback<Build<{ setSlice: (value: number) => void }>>((renderer, own) => {
    const image = own(vtkImageData.newInstance());
    image.setDimensions(...config.dimensions);
    image.setSpacing(config.spacing);
    const values = new Float32Array(field.values.length);
    for (let i = 0; i < values.length; i++) values[i] = field.mask[i] ? scalePosition(field.values[i], scale) : MASKED_POSITION;
    image.getPointData().setScalars(own(vtkDataArray.newInstance({ name: 'field', numberOfComponents: 1, values })));
    const lookup = lookupTable(scale, own);
    const mappers = [SlicingMode.I, SlicingMode.J, SlicingMode.K].map((mode, axis) => {
      const mapper = own(vtkImageMapper.newInstance());
      mapper.setInputData(image);
      mapper.setSlicingMode(mode);
      mapper.setSlice(axis === 2 ? currentSlice.current : Math.floor((config.dimensions[axis] - 1) / 2));
      const actor = own(vtkImageSlice.newInstance());
      actor.setMapper(mapper);
      actor.getProperty().setRGBTransferFunction(0, lookup);
      actor.getProperty().setUseLookupTableScalarRange(true);
      actor.getProperty().setInterpolationTypeToNearest();
      renderer.addViewProp(actor);
      return mapper;
    });
    return { setSlice: (value: number) => { mappers[2].setSlice(value); renderer.getRenderWindow()?.render(); } };
  }, [field, config, scale]);
  const { container, scene, error } = useScene(config, build);
  useEffect(() => scene.current?.controls.setSlice(slice), [scene, slice]);
  return <SceneView container={container} error={error} label="Interactive 3D orthogonal field slices" onReset={() => scene.current?.reset()} />;
}

/** Boundary faces colored by one surface channel, faces facing the camera are culled so the inside stays visible from any angle. */
export function Surface({ surface, channel, config, scale }: { surface: SurfaceData; channel: number; config: Config; scale: ColorScale }) {
  const build = useCallback<Build<null>>((renderer, own) => {
    const polyData = own(vtkPolyData.newInstance());
    polyData.getPoints().setData(surface.points, 3);
    polyData.getPolys().setData(surface.polygons);
    const positions = surface.values[channel].map((value) => Number.isNaN(value) ? NaN : scalePosition(value, scale));
    polyData.getCellData().setScalars(own(vtkDataArray.newInstance({ name: 'surface', numberOfComponents: 1, values: positions })));
    const mapper = own(vtkMapper.newInstance());
    mapper.setInputData(polyData);
    mapper.setLookupTable(lookupTable(scale, own));
    mapper.setUseLookupTableScalarRange(true);
    mapper.setScalarModeToUseCellData();
    const actor = own(vtkActor.newInstance());
    actor.setMapper(mapper);
    actor.getProperty().setFrontfaceCulling(true);
    actor.getProperty().setAmbient(0.6);
    actor.getProperty().setDiffuse(0.4);
    renderer.addActor(actor);
    return null;
  }, [surface, channel, scale]);
  const { container, scene, error } = useScene(config, build);
  return <SceneView container={container} error={error} label="Interactive 3D boundary surfaces" onReset={() => scene.current?.reset()} />;
}
