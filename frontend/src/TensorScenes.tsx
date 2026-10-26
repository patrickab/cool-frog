import { Canvas, useFrame, useThree } from '@react-three/fiber';
import type { ThreeEvent } from '@react-three/fiber';
import { Fragment, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { Config, Field } from './api';
import { colorAt, oklchCss, oklchToLinear, themeColors } from './lib/colormap';
import type { Axis, ColorScale } from './lib/colormap';
import { Disclosure, Panel, Segmented, Slider, useTheme } from './lib/ui';

/** Exact integer with thousands separators, value counts here must not be rounded. */
const formatCount = (value: number) => value.toLocaleString('en-US');

/* ---------- Tadpole configuration, `tadpole.architecture.p3d` and paper Table 12 ---------- */

export type Size = 'S' | 'B' | 'L';
/** `hidden` = transformer width at 16³, latent channels are `4 · hidden`, `features` = conv widths (FED). */
const SIZES: Record<Size, { hidden: number; features: readonly [number, number, number]; heads: number; groups: number; compression: number; parameters: string }> = {
  S: { hidden: 64, features: [32, 32, 64], heads: 4, groups: 16, compression: 16, parameters: '3.4 + 5.4 M' },
  B: { hidden: 128, features: [64, 128, 128], heads: 4, groups: 32, compression: 8, parameters: '15.0 + 23.1 M' },
  L: { hidden: 256, features: [128, 256, 256], heads: 8, groups: 32, compression: 4, parameters: '59.9 + 92.2 M' },
};
const CROP = 64;
const DOWNSAMPLING = 16;
const WINDOW = 4;
const latentChannels = (size: Size) => 4 * SIZES[size].hidden;
/** Size every scene shows, the paper's headline model. */
const SIZE: Size = 'B';

/** Pillar height for a channel count: square root keeps 1 to 2,048 channels on one screen. */
const pillarHeight = (channels: number) => 0.86 * Math.max(1, Math.sqrt(channels) / 4);

/* ---------- Stages of one forward pass, shapes traced with forward hooks on S, B and L ---------- */

type Arrangement = 'volume' | 'crops' | 'batch' | 'downstream';
type Part = 'Rearrange' | 'Encoder' | 'KL head' | 'Decoder' | 'Downstream';
type Stage = {
  title: string; part: Part; detail: (size: Size) => string; arrangement: Arrangement;
  /** Grid divisor per crop axis against the 64³ input crop. */
  divisor: number;
  channels: (size: Size) => number;
  kl?: 'split' | 'sample';
};
const hidden = (factor: number) => (size: Size) => SIZES[size].hidden * factor;
const feature = (index: 0 | 1) => (size: Size) => SIZES[size].features[index];
const one = () => 1;
const attention = (size: Size) => `2 blocks, ${SIZES[size].heads} heads, ${WINDOW}³ windows`;
const STAGES: Stage[] = [
  { title: 'Input', part: 'Rearrange', detail: () => 'Fields', arrangement: 'volume', divisor: 1, channels: one },
  { title: 'Cut into crops', part: 'Rearrange', detail: () => 'Split every axis', arrangement: 'crops', divisor: 1, channels: one },
  { title: 'Fold into batch', part: 'Rearrange', detail: () => 'C and crops join B', arrangement: 'batch', divisor: 1, channels: one },
  { title: 'Embed', part: 'Encoder', detail: () => '3³ conv', arrangement: 'batch', divisor: 1, channels: feature(0) },
  { title: 'Conv ↓2', part: 'Encoder', detail: () => '3³ conv stride 2, 2 residual blocks', arrangement: 'batch', divisor: 2, channels: feature(1) },
  { title: 'Conv ↓2', part: 'Encoder', detail: () => '3³ conv stride 2', arrangement: 'batch', divisor: 4, channels: hidden(1) },
  { title: 'Attention', part: 'Encoder', detail: attention, arrangement: 'batch', divisor: 4, channels: hidden(1) },
  { title: 'Conv ↓2', part: 'Encoder', detail: () => '3³ conv stride 2', arrangement: 'batch', divisor: 8, channels: hidden(2) },
  { title: 'Attention', part: 'Encoder', detail: attention, arrangement: 'batch', divisor: 8, channels: hidden(2) },
  { title: 'Conv ↓2', part: 'Encoder', detail: () => '3³ conv stride 2', arrangement: 'batch', divisor: 16, channels: hidden(4) },
  { title: 'Attention', part: 'Encoder', detail: attention, arrangement: 'batch', divisor: 16, channels: hidden(4) },
  { title: 'μ, log σ²', part: 'KL head', detail: () => '1×1 conv, split in two', arrangement: 'batch', divisor: 16, channels: hidden(8), kl: 'split' },
  { title: 'Sample z', part: 'KL head', detail: () => 'z = μ + σ · ε', arrangement: 'batch', divisor: 16, channels: hidden(4), kl: 'sample' },
  { title: 'Upsample', part: 'Decoder', detail: () => '3³ conv, pixel shuffle, 3³ conv', arrangement: 'batch', divisor: 8, channels: hidden(2) },
  { title: 'Attention', part: 'Decoder', detail: attention, arrangement: 'batch', divisor: 8, channels: hidden(2) },
  { title: 'Upsample', part: 'Decoder', detail: () => '3³ conv, pixel shuffle, 3³ conv', arrangement: 'batch', divisor: 4, channels: hidden(1) },
  { title: 'Attention', part: 'Decoder', detail: (size) => `${attention(size)}, norm, 3³ conv`, arrangement: 'batch', divisor: 4, channels: hidden(1) },
  { title: 'Upsample', part: 'Decoder', detail: () => '1×1 conv, pixel shuffle, 2 residual blocks', arrangement: 'batch', divisor: 2, channels: feature(1) },
  { title: 'Upsample', part: 'Decoder', detail: () => '1×1 conv, pixel shuffle', arrangement: 'batch', divisor: 1, channels: feature(0) },
  { title: 'Output conv', part: 'Decoder', detail: () => '3³ conv', arrangement: 'batch', divisor: 1, channels: one },
  { title: 'Inverse rearrange', part: 'Rearrange', detail: () => 'Crops back in place', arrangement: 'volume', divisor: 1, channels: one },
  { title: 'Latent layout', part: 'Downstream', detail: () => 'Tadpole-DFT input', arrangement: 'downstream', divisor: 16, channels: hidden(4) },
];
/** Stage ranges of the explorers: rearrange only, the full autoencoder, sampled latent to downstream layout. */
const EXPLORER_STAGES = {
  rearrange: [0, 1, 2],
  network: STAGES.slice(2, 21).map((_, index) => index + 2),
  layout: [12, 21],
};

/* ---------- Toy tensor: one persistent element per input voxel ---------- */

const TOY_EDGE = 8;
type Toy = { batch: number; channels: number; crop: number; factor: number };
type Element = { b: number; c: number; u: number; v: number; w: number; i: number; j: number; k: number };
type Coordinate = 'b' | 'c' | 'u' | 'v' | 'w' | 'ci' | 'cj' | 'ck' | 'q';

function toyElements({ batch, channels, crop }: Toy): Element[] {
  const elements: Element[] = [];
  for (let b = 0; b < batch; b++) for (let c = 0; c < channels; c++)
    for (let x = 0; x < TOY_EDGE; x++) for (let y = 0; y < TOY_EDGE; y++) for (let z = 0; z < TOY_EDGE; z++)
      elements.push({ b, c, u: Math.floor(x / crop), v: Math.floor(y / crop), w: Math.floor(z / crop), i: x % crop, j: y % crop, k: z % crop });
  return elements;
}

const TOY_VALUE_SCALE: ColorScale = { colormap: 'sequential', range: [0, 1], log: false };
/** Toy field number `field` = b · C + c: a smooth blob whose center moves with the field. */
const toyValue = (field: number, x: number, y: number, z: number) => Math.exp(-(
  (x - 3.5 - 1.5 * Math.cos(2.4 * field)) ** 2 + (y - 3.5 - 1.5 * Math.sin(2.4 * field)) ** 2 + (z - 3.5 - Math.cos(1.3 * field)) ** 2) / 8);

/** Toy cell edge for a real divisor: the toy `factor` stands for Tadpole's 16, intermediate stages scale log-linearly. */
const toyGroup = (divisor: number, factor: number) => 2 ** Math.floor((Math.log2(divisor) * Math.log2(factor)) / Math.log2(DOWNSAMPLING));

function coordinate(element: Element, name: Coordinate, group: number): number {
  if (name === 'ci') return Math.floor(element.i / group);
  if (name === 'cj') return Math.floor(element.j / group);
  if (name === 'ck') return Math.floor(element.k / group);
  if (name === 'q') return ((element.i % group) * group + (element.j % group)) * group + (element.k % group);
  return element[name];
}

/** Index of the element's crop within its field, the order `rearrange` flattens U, V, W in. */
const cropIndex = (element: Element, cropsPerAxis: number) => (element.u * cropsPerAxis + element.v) * cropsPerAxis + element.w;

const HIDDEN_SCALE = 1e-3;

function arrange(stage: Stage, toy: Toy, size: Size, elements: Element[]): { positions: Float32Array; scales: Float32Array } {
  const { batch, channels, crop, factor } = toy;
  const cropsPerAxis = TOY_EDGE / crop;
  const positions = new Float32Array(elements.length * 3);
  const scales = new Float32Array(elements.length * 3).fill(0.86);
  const group = stage.arrangement === 'downstream' ? factor : toyGroup(stage.divisor, factor);
  const cells = crop / group;
  const set = (index: number, x: number, y: number, z: number, scale?: [number, number, number]) => {
    positions.set([x, y, z], index * 3);
    if (scale) scales.set(scale, index * 3);
  };
  elements.forEach((element, index) => {
    const { b, c, u, v, w, i, j, k } = element;
    const field = b * channels + c;
    if (stage.arrangement === 'volume' || stage.arrangement === 'crops') {
      const gap = stage.arrangement === 'crops' ? 1 : 0;
      const width = TOY_EDGE + (cropsPerAxis - 1) * gap + 2;
      set(index, field * width + b * 2 + u * (crop + gap) + i, w * (crop + gap) + k, v * (crop + gap) + j);
      return;
    }
    const q = coordinate(element, 'q', group);
    const [ci, cj, ck] = (['ci', 'cj', 'ck'] as const).map((name) => coordinate(element, name, group));
    if (stage.arrangement === 'downstream') {
      const pillar = pillarHeight(stage.channels(size));
      const pitch = pillar + 0.8;
      const spacing = 1.8;
      const width = 0.86 / channels;
      set(index, b * (cropsPerAxis * cells * spacing + 2) + (u * cells + ci) * spacing + (c - (channels - 1) / 2) * width, (w * cells + ck) * pitch + pillar / 2, (v * cells + cj) * spacing,
        q === 0 ? [width * 0.9, pillar, 0.86] : [HIDDEN_SCALE, HIDDEN_SCALE, HIDDEN_SCALE]);
      return;
    }
    const split = stage.kl === 'split';
    const pillar = pillarHeight(stage.channels(size) / (split ? 2 : 1));
    const pitch = Math.max(1, pillar + 0.14);
    const center = (crop - 1) / 2;
    const x = cropIndex(element, cropsPerAxis) * (crop + 1.5) + center + ci - (cells - 1) / 2;
    const z = field * (crop + 1.5) + center + cj - (cells - 1) / 2;
    const y = ck * pitch + pillar / 2;
    if (split && q < 2) set(index, x + (q ? 0.27 : -0.27), y, z, [0.34, pillar, 0.86]);
    else if (q === 0) set(index, x, y, z, [0.86, pillar, 0.86]);
    else set(index, x, y, z, [HIDDEN_SCALE, HIDDEN_SCALE, HIDDEN_SCALE]);
  });
  return { positions, scales };
}

/* ---------- Shared 3D primitives ---------- */

type Palette = ReturnType<typeof themeColors>;
/** Mix a linear color toward the surface, 0 keeps it, 1 is the surface. */
const dim = (color: readonly number[], amount: number, surface: readonly number[]) => color.map((component, index) => component + (surface[index] - component) * amount);
const linearColor = (rgb: readonly number[]) => COLOR.setRGB(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, THREE.SRGBColorSpace).toArray();
/** Learned features are neutral, with contrast measured from solid toward ink in either theme. */
const featureGray = (palette: Palette, contrast: number, chroma = palette.solid[1]) => oklchToLinear([
  palette.solid[0] + Math.sign(palette.ink[0] - palette.solid[0]) * contrast, chroma, palette.solid[2],
]);

const MATRIX = new THREE.Matrix4();
const COLOR = new THREE.Color();
const BOX = new THREE.BoxGeometry(1, 1, 1);
const EDGES = new THREE.EdgesGeometry(BOX);
const VIEW_DIRECTION = new THREE.Vector3(0.9, 1, 1.5).normalize();
const TRANSITION_MS = 700;
const SCENE_CAMERA = { near: 1, far: 4000, position: [0, 0, 1000] as THREE.Vector3Tuple };

/** Instanced cubes, position and scale changes animate so every element can be followed, colors switch at once. */
function Voxels({ positions, scales, colors, onHover, onSelect }: {
  positions: Float32Array; scales: Float32Array; colors: Float32Array;
  onHover?: (index: number | undefined) => void; onSelect?: (index: number) => void;
}) {
  const count = positions.length / 3;
  const mesh = useRef<THREE.InstancedMesh>(null);
  const invalidate = useThree((state) => state.invalidate);
  const shown = useRef<{ positions: Float32Array; scales: Float32Array } | null>(null);
  const transition = useRef<{ positions: Float32Array; scales: Float32Array; start: number } | null>(null);
  const write = (shownPositions: Float32Array, shownScales: Float32Array) => {
    const instances = mesh.current!;
    for (let index = 0; index < count; index++) {
      MATRIX.makeScale(shownScales[index * 3], shownScales[index * 3 + 1], shownScales[index * 3 + 2]);
      MATRIX.setPosition(shownPositions[index * 3], shownPositions[index * 3 + 1], shownPositions[index * 3 + 2]);
      instances.setMatrixAt(index, MATRIX);
    }
    instances.instanceMatrix.needsUpdate = true;
    instances.boundingSphere = null;
    instances.boundingBox = null;
  };
  useLayoutEffect(() => {
    if (!shown.current || shown.current.positions.length !== positions.length) {
      shown.current = { positions: positions.slice(), scales: scales.slice() };
      transition.current = null;
      write(positions, scales);
    } else {
      transition.current = { positions: shown.current.positions.slice(), scales: shown.current.scales.slice(), start: performance.now() };
    }
    invalidate();
  }, [positions, scales]); // eslint-disable-line react-hooks/exhaustive-deps
  useLayoutEffect(() => {
    const instances = mesh.current!;
    for (let index = 0; index < count; index++) instances.setColorAt(index, COLOR.setRGB(colors[index * 3], colors[index * 3 + 1], colors[index * 3 + 2]));
    instances.instanceColor!.needsUpdate = true;
    invalidate();
  }, [colors, count, invalidate]);
  useFrame(() => {
    const from = transition.current;
    if (!from || !shown.current) return;
    const progress = Math.min(1, (performance.now() - from.start) / TRANSITION_MS);
    const eased = progress < 0.5 ? 2 * progress * progress : 1 - (-2 * progress + 2) ** 2 / 2;
    for (let index = 0; index < positions.length; index++) {
      shown.current.positions[index] = from.positions[index] + (positions[index] - from.positions[index]) * eased;
      shown.current.scales[index] = from.scales[index] + (scales[index] - from.scales[index]) * eased;
    }
    write(shown.current.positions, shown.current.scales);
    if (progress < 1) invalidate();
    else transition.current = null;
  });
  const hover = (event: ThreeEvent<PointerEvent>) => { event.stopPropagation(); onHover?.(event.instanceId); };
  return <instancedMesh key={count} ref={mesh} args={[BOX, undefined, count]}
    onPointerMove={onHover && hover} onPointerOut={onHover && (() => onHover(undefined))}
    onClick={onSelect && ((event) => { event.stopPropagation(); if (event.instanceId !== undefined) onSelect(event.instanceId); })}>
    <meshLambertMaterial />
  </instancedMesh>;
}

/** Box outline from `min` to `max` corner, in ink. */
function Outline({ min, max }: { min: THREE.Vector3Tuple; max: THREE.Vector3Tuple }) {
  const { theme } = useTheme();
  const ink = themeColors(theme).ink;
  const color = useMemo(() => new THREE.Color(...oklchToLinear(ink)), [ink]);
  return <lineSegments geometry={EDGES} position={min.map((value, axis) => (value + max[axis]) / 2) as THREE.Vector3Tuple}
    scale={min.map((value, axis) => max[axis] - value) as THREE.Vector3Tuple}>
    <lineBasicMaterial color={color} />
  </lineSegments>;
}

/** Straight segments between consecutive point pairs. */
function Segments({ points, opacity = 1 }: { points: number[]; opacity?: number }) {
  const { theme } = useTheme();
  const ink = themeColors(theme).ink;
  const color = useMemo(() => new THREE.Color(...oklchToLinear(ink)), [ink]);
  const geometry = useMemo(() => new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(points, 3)), [points]);
  return <lineSegments geometry={geometry} renderOrder={1}>
    <lineBasicMaterial color={color} transparent opacity={opacity} depthTest={false} />
  </lineSegments>;
}

/** Fixed orthographic framing of `bounds`, recomputed only when the bounds or canvas size change. */
function Fit({ bounds }: { bounds: THREE.Box3 }) {
  const { camera, size, controls, invalidate } = useThree();
  useLayoutEffect(() => {
    const center = bounds.getCenter(new THREE.Vector3());
    camera.position.copy(center).addScaledVector(VIEW_DIRECTION, 1000);
    camera.lookAt(center);
    camera.updateMatrixWorld();
    const centerView = center.clone().applyMatrix4(camera.matrixWorldInverse);
    let halfWidth = 1;
    let halfHeight = 1;
    for (const corner of [0, 1, 2, 3, 4, 5, 6, 7]) {
      const point = new THREE.Vector3(corner & 1 ? bounds.max.x : bounds.min.x, corner & 2 ? bounds.max.y : bounds.min.y, corner & 4 ? bounds.max.z : bounds.min.z)
        .applyMatrix4(camera.matrixWorldInverse);
      halfWidth = Math.max(halfWidth, Math.abs(point.x - centerView.x));
      halfHeight = Math.max(halfHeight, Math.abs(point.y - centerView.y));
    }
    (camera as THREE.OrthographicCamera).zoom = 0.94 * Math.min(size.width / (2 * halfWidth), size.height / (2 * halfHeight));
    camera.updateProjectionMatrix();
    const orbit = controls as unknown as { target: THREE.Vector3; update: () => void } | null;
    orbit?.target.copy(center);
    orbit?.update();
    invalidate();
  }, [bounds, camera, size.width, size.height, controls, invalidate]);
  return null;
}

/** Orbit on drag, registered as the default controls so `Fit` can retarget them. */
function Orbit() {
  const { camera, gl, invalidate, set } = useThree();
  useEffect(() => {
    const controls = new OrbitControls(camera, gl.domElement);
    controls.addEventListener('change', () => invalidate());
    set({ controls });
    return () => { controls.dispose(); set({ controls: null }); };
  }, [camera, gl, invalidate, set]);
  return null;
}

/** Canvas on the panel surface: soft light, orbit on drag, no animation loop. */
function Scene({ bounds, label, height = 380, children }: { bounds: THREE.Box3; label: string; height?: number; children: ReactNode }) {
  const { theme } = useTheme();
  const surface = themeColors(theme).surface;
  const background = useMemo(() => oklchToLinear(surface), [surface]);
  return <div role="img" aria-label={label} style={{ height }} className="cursor-grab active:cursor-grabbing">
    <Canvas orthographic frameloop="demand" dpr={[1, 2]} camera={SCENE_CAMERA} gl={{ antialias: true }}>
      <color attach="background" args={background} />
      <ambientLight intensity={2.1} />
      <directionalLight position={[3, 10, 6]} intensity={1.4} />
      {children}
      <Orbit />
      <Fit bounds={bounds} />
    </Canvas>
  </div>;
}

function boundsOf(layouts: { positions: Float32Array; scales: Float32Array }[]): THREE.Box3 {
  const box = new THREE.Box3();
  const point = new THREE.Vector3();
  for (const { positions, scales } of layouts)
    for (let index = 0; index < positions.length; index += 3) {
      if (scales[index] <= HIDDEN_SCALE) continue;
      for (const sign of [-0.5, 0.5]) box.expandByPoint(point.set(
        positions[index] + sign * scales[index], positions[index + 1] + sign * scales[index + 1], positions[index + 2] + sign * scales[index + 2]));
    }
  return box;
}

/* ---------- Shape notation ---------- */

type AxisSymbol = { text: string; axis: Axis; varies: Coordinate[] };
type Dimension = AxisSymbol[];

/** `H_X/4` renders as H with subscript X, then /4. */
function Notation({ text }: { text: string }) {
  const [, base, subscript, rest] = text.match(/^([^_]*)(?:_([A-Za-z]+))?(.*)$/)!;
  return <>{base}{subscript && <sub>{subscript}</sub>}{rest}</>;
}

function shapeOf(stage: Stage, size: Size): Dimension[] {
  const symbol = (text: string, axis: Axis, varies: Coordinate[] = []): AxisSymbol => ({ text, axis, varies });
  const batch = symbol('B', 'batch', ['b']);
  const channel = symbol('C', 'channel', ['c']);
  const latent = stage.divisor === DOWNSAMPLING;
  const space = (['X', 'Y', 'Z'] as const).map((name, axis) => ({ name, crop: (['u', 'v', 'w'] as const)[axis], cell: (['ci', 'cj', 'ck'] as const)[axis] }));
  const cropSpace = (name: string, cell: Coordinate) => symbol(latent ? `${name}_l` : stage.divisor > 1 ? `H_${name}/${stage.divisor}` : `H_${name}`, 'spatial', [cell]);
  const channels = stage.channels(size);
  const features: Dimension = latent && stage.kl === 'split' ? [symbol('2', 'feature'), symbol('C_l', 'feature', ['q'])]
    : latent ? [symbol('C_l', 'feature', ['q'])] : [symbol(String(channels), 'feature', channels > 1 ? ['q'] : [])];
  if (stage.arrangement === 'volume') return [[batch], [channel], ...space.map(({ name, crop, cell }) => [symbol(name, 'spatial', [crop, cell])])];
  if (stage.arrangement === 'crops') return [[batch], [channel], ...space.map(({ name, crop, cell }) => [symbol(name.replace('X', 'U').replace('Y', 'V').replace('Z', 'W'), 'crop', [crop]), cropSpace(name, cell)])];
  if (stage.arrangement === 'downstream') return [[batch], [channel, symbol('C_l', 'feature', ['q'])], ...space.map(({ name, crop, cell }) => [symbol(name.replace('X', 'U').replace('Y', 'V').replace('Z', 'W'), 'crop', [crop]), cropSpace(name, cell)])];
  return [[batch, channel, symbol('N', 'crop', ['u', 'v', 'w'])], features, ...space.map(({ name, cell }) => [cropSpace(name, cell)])];
}

/** Size of one symbol in the toy tensor. */
function symbolValue(symbol: AxisSymbol, stage: Stage, toy: Toy): number {
  const cropsPerAxis = TOY_EDGE / toy.crop;
  if (symbol.text === 'B') return toy.batch;
  if (symbol.text === 'C') return toy.channels;
  if (symbol.text === 'N') return cropsPerAxis ** 3;
  if ('UVW'.includes(symbol.text)) return cropsPerAxis;
  if (symbol.text === 'C_l') return latentChannels(SIZE);
  if ('XYZ'.includes(symbol.text)) return cropsPerAxis * toy.crop;
  if (symbol.text.endsWith('_l')) return toy.crop / toy.factor;
  if (symbol.text.startsWith('H_')) return toy.crop / toyGroup(stage.divisor, toy.factor);
  return Number(symbol.text);
}

/* ---------- Explorer: one toy tensor through every stage ---------- */

type Highlight = { varies: Coordinate[] } | undefined;

/** Field values in viridis while a stage holds values, gray once it holds learned features. */
function elementColors(elements: Element[], toy: Toy, stage: Stage, selected: number | undefined, highlight: Highlight, palette: Palette): Float32Array {
  const surface = oklchToLinear(palette.surface);
  const fields = toy.batch * toy.channels;
  const group = stage.arrangement === 'downstream' ? toy.factor : toyGroup(stage.divisor, toy.factor);
  const showsValues = stage.channels(SIZE) === 1;
  const fixed: Coordinate[] = ['b', 'c', 'u', 'v', 'w', 'ci', 'cj', 'ck', 'q'];
  const anchor = selected === undefined ? undefined : elements[selected];
  const colors = new Float32Array(elements.length * 3);
  elements.forEach((element, index) => {
    const field = element.b * toy.channels + element.c;
    const logVariance = stage.kl === 'split' && coordinate(element, 'q', group) === 1;
    let color: readonly number[] = showsValues
      ? linearColor(colorAt(toyValue(field, element.u * toy.crop + element.i, element.v * toy.crop + element.j, element.w * toy.crop + element.k), TOY_VALUE_SCALE))
      : featureGray(palette, 0.16 + (0.16 * field) / Math.max(1, fields - 1) - (logVariance ? 0.12 : 0), palette.solid[1] * 5 / 6);
    if (anchor) {
      const keep = highlight ? fixed.filter((name) => !highlight.varies.includes(name)) : fixed.filter((name) => name !== 'q');
      const matches = keep.every((name) => coordinate(element, name, group) === coordinate(anchor, name, group));
      if (!matches) color = dim(color, highlight ? 0.85 : 0.6, surface);
    }
    colors.set(color, index * 3);
  });
  return colors;
}


/** Shape symbols with their toy sizes, hovering a symbol highlights the cubes along its axis. */
function ShapeReadout({ stage, toy, onHighlight }: { stage: Stage; toy: Toy; onHighlight: (highlight: Highlight) => void }) {
  const { theme } = useTheme();
  const palette = themeColors(theme);
  const shape = shapeOf(stage, SIZE);
  const tuple = (render: (symbol: AxisSymbol) => ReactNode) => <>({shape.map((dimension, index) => <Fragment key={index}>{index > 0 && ', '}
    {dimension.map((symbol, part) => <Fragment key={part}>{part > 0 && '·'}{render(symbol)}</Fragment>)}</Fragment>)})</>;
  return <p className="text-base tabular-nums">
    {tuple((symbol) => <button type="button" title="Highlight this axis" style={{ color: oklchCss(palette.axes[symbol.axis]) }}
      className="rounded px-px font-medium hover:bg-hover-strong focus-visible:bg-hover-strong focus-visible:outline-none"
      onPointerEnter={() => onHighlight({ varies: symbol.varies })} onPointerLeave={() => onHighlight(undefined)}
      onFocus={() => onHighlight({ varies: symbol.varies })} onBlur={() => onHighlight(undefined)}><Notation text={symbol.text} /></button>)}
    <span className="text-ink-muted"> = {tuple((symbol) => symbolValue(symbol, stage, toy))}</span>
  </p>;
}

function StageDiagram({ stages, active, size, onSelect }: { stages: number[]; active: number; size: Size; onSelect: (stage: number) => void }) {
  const groups: { part: Part; indices: number[] }[] = [];
  for (const index of stages) {
    const previous = groups.at(-1);
    if (previous?.part === STAGES[index].part) previous.indices.push(index);
    else groups.push({ part: STAGES[index].part, indices: [index] });
  }
  return <div className="space-y-2">
    {groups.map(({ part, indices }) => <div key={indices[0]} className="flex flex-wrap items-center gap-1.5 text-[11px]">
      <span className="w-20 shrink-0 text-ink-muted">{part}</span>
      {indices.map((index, position) => {
        const stage = STAGES[index];
        const edge = CROP / stage.divisor;
        return <Fragment key={index}>
          {position > 0 && <span aria-hidden className="text-ink-faint">→</span>}
          <button type="button" onClick={() => onSelect(index)} aria-pressed={index === active} title={stage.detail(size)}
            className={`rounded-lg px-2 py-1 text-left ${index === active ? 'bg-ink text-paper' : 'bg-paper hover:bg-hover-strong'}`}>
            <span className="block font-medium">{stage.title}</span>
            <span className={`block tabular-nums ${index === active ? 'text-paper/80' : 'text-ink-muted'}`}>
              {stage.arrangement === 'batch' ? `${stage.channels(size)} × ${edge}³` : stage.arrangement === 'downstream' ? `C·${latentChannels(size)} × ${edge}³` : `1 × ${CROP}³`}
            </span>
          </button>
        </Fragment>;
      })}
    </div>)}
  </div>;
}

const COUNT_OPTIONS = [1, 2, 3].map((value) => ({ value, label: String(value) }));

/** Plain-language headline and one sentence per stage, shown above the scene. */
function stageLesson(stage: Stage): { action: string; explanation: string } {
  if (stage.arrangement === 'downstream') return {
    action: 'Bring the variables back together',
    explanation: 'Crop latents return to their places, the variables stack at each position. Values are unchanged.',
  };
  if (stage.part === 'Rearrange') {
    if (stage.title === 'Inverse rearrange') return {
      action: 'Put the reconstructed crops back',
      explanation: 'One volume per variable again, now holding the decoder’s values.',
    };
    if (stage.arrangement === 'volume') return {
      action: 'Start with separate scalar fields',
      explanation: 'Each volume is one variable of one sample.',
    };
    if (stage.arrangement === 'crops') return {
      action: 'Separate the spatial crops',
      explanation: 'Same values, only moved apart.',
    };
    return {
      action: 'Give every crop its own batch entry',
      explanation: 'Each crop is now an independent input to the same encoder.',
    };
  }
  if (stage.kl === 'split') return {
    action: 'Describe a distribution at each latent position',
    explanation: 'Two pillars per position: a mean and a log-variance.',
  };
  if (stage.kl === 'sample') return {
    action: 'Sample the latent',
    explanation: 'z = μ + σ · ε. Everything the decoder gets passes through here.',
  };
  if (stage.title === 'Attention') return {
    action: 'Mix information without shrinking the grid',
    explanation: 'Features change, the shape does not.',
  };
  if (stage.part === 'Decoder') return stage.title === 'Output conv'
    ? { action: 'Predict one value per position', explanation: 'A last convolution turns the features into the reconstructed field.' }
    : { action: 'Turn features back into spatial detail', explanation: 'Pixel shuffle moves 8 channels into a 2 × 2 × 2 block, each axis doubles.' };
  return stage.title === 'Embed'
    ? { action: 'Replace each value with learned features', explanation: 'Same grid, more features per position.' }
    : { action: 'Halve the grid', explanation: 'A stride-2 convolution: one eighth the positions, more features each.' };
}

/**
 * Toy tensor animated through `stages`, with Tadpole-B sizes.
 *
 * - `diagram` shows the flat block diagram of the selected stages, clicking a block jumps there
 * - Toy edge 8, crop 4 or 8 and factor 4 stand in for Tadpole's 64³ crops and factor 16
 */
export function TensorExplorer({ range, diagram = false, label }: { range: keyof typeof EXPLORER_STAGES; diagram?: boolean; label: string }) {
  const { theme } = useTheme();
  const palette = themeColors(theme);
  const stages = EXPLORER_STAGES[range];
  const [fields, setFields] = useState({ batch: 1, channels: range === 'network' ? 1 : 2 });
  const toy = useMemo<Toy>(() => ({ ...fields, crop: range === 'network' ? 8 : 4, factor: 4 }), [fields, range]);
  const [position, setPosition] = useState(0);
  const [selected, setSelected] = useState<number | undefined>(undefined);
  const [highlight, setHighlight] = useState<Highlight>(undefined);
  const elements = useMemo(() => toyElements(toy), [toy]);
  const layouts = useMemo(() => stages.map((index) => {
    const layout = arrange(STAGES[index], toy, SIZE, elements);
    if (range === 'network') {
      // Keep shrinking grids centered while the camera and world scale stay fixed.
      const extent = boundsOf([layout]);
      const centerY = (extent.min.y + extent.max.y) / 2;
      for (let coordinate = 1; coordinate < layout.positions.length; coordinate += 3) layout.positions[coordinate] -= centerY;
    }
    return layout;
  }), [stages, toy, elements, range]);
  const bounds = useMemo(() => boundsOf(layouts), [layouts]);
  const stage = STAGES[stages[position]];
  const lesson = stageLesson(stage);
  const anchor = selected ?? (highlight ? 0 : undefined);
  const colors = useMemo(() => elementColors(elements, toy, stage, anchor, highlight, palette), [elements, toy, stage, anchor, highlight, palette]);
  const updateFields = (next: Partial<typeof fields>) => {
    setFields({ ...fields, ...next });
    setSelected(undefined);
  };
  const edge = CROP / stage.divisor;
  const channels = stage.channels(SIZE);
  const valueRatio = (edge ** 3 * channels) / CROP ** 3;
  return <Panel className="space-y-4 p-4">
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div className="max-w-2xl space-y-1">
        <p className="text-xs text-ink-muted">{stage.part} · {position + 1} / {stages.length}</p>
        <h3 className="text-base font-medium">{lesson.action}</h3>
        <p className="text-sm leading-relaxed text-ink-muted">{lesson.explanation}</p>
      </div>
      <div className="flex gap-2">
        <button type="button" disabled={position === 0} onClick={() => setPosition(position - 1)} className="rounded-lg bg-paper px-3 py-2 text-xs hover:bg-hover-strong disabled:opacity-40">← Previous</button>
        <button type="button" disabled={position === stages.length - 1} onClick={() => setPosition(position + 1)} className="rounded-lg bg-ink px-3 py-2 text-xs text-paper disabled:opacity-40">Next →</button>
      </div>
    </div>
    {stage.arrangement === 'batch' && <p className="text-sm tabular-nums">
      Tadpole-{SIZE}: <strong>{edge}³</strong> positions × <strong>{channels}</strong> {stage.kl === 'split' ? 'parameters' : channels === 1 ? 'value' : 'features'}
      {valueRatio !== 1 && <span className="text-ink-muted">, {valueRatio > 1 ? `${formatCount(valueRatio)}× more` : `${formatCount(1 / valueRatio)}× fewer`} values than the crop</span>}
    </p>}
    <figure className="space-y-2">
      <Scene bounds={bounds} label={label}>
        <Voxels {...layouts[position]} colors={colors} onSelect={(index) => setSelected(index === selected ? undefined : index)} />
      </Scene>
      {range !== 'rearrange' && <figcaption className="text-xs text-ink-muted">Colored = field value · gray = learned features</figcaption>}
    </figure>
    <div className="space-y-1.5">
      <p className="text-xs"><span className="font-medium">{stage.title}</span> <span className="text-ink-muted">{stage.detail(SIZE)}</span></p>
      <Slider label="Stage" min={0} max={stages.length - 1} step={1} values={[position]} onChange={([next]) => setPosition(next)} />
    </div>
    {range !== 'network' && <div className="flex flex-wrap items-center gap-x-6 gap-y-3 text-xs">
      <span className="flex items-center gap-2">Samples (B) <Segmented label="Samples" value={fields.batch} onChange={(batch) => updateFields({ batch })} options={COUNT_OPTIONS} /></span>
      <span className="flex items-center gap-2">Variables (C) <Segmented label="Variables per sample" value={fields.channels} onChange={(channels) => updateFields({ channels })} options={COUNT_OPTIONS} /></span>
    </div>}
    {range === 'rearrange' && <ShapeReadout stage={stage} toy={toy} onHighlight={setHighlight} />}
    {diagram && <Disclosure label="Full network · select a layer"><div className="pt-3"><StageDiagram stages={stages} active={stages[position]} size={SIZE} onSelect={(index) => setPosition(stages.indexOf(index))} /></div></Disclosure>}
  </Panel>;
}

/** Configuration of every size. */
export function SizeTable() {
  const rows: [string, (value: Size) => ReactNode][] = [
    ['Conv widths (FED)', (value) => SIZES[value].features.join(', ')],
    ['Transformer widths', (value) => [1, 2, 4].map((factor) => SIZES[value].hidden * factor).join(', ')],
    ['Attention heads', (value) => SIZES[value].heads],
    ['GroupNorm groups', (value) => SIZES[value].groups],
    ['Latent channels C_l', (value) => latentChannels(value)],
    ['Compression 64³ / (C_l · 4³)', (value) => `${SIZES[value].compression}×`],
    ['Parameters, encoder + decoder', (value) => SIZES[value].parameters],
  ];
  return <Panel className="max-w-3xl overflow-x-auto">
    <table className="w-full text-left text-xs tabular-nums">
      <caption className="sr-only">Tadpole configurations</caption>
      <thead className="shadow-[inset_0_-1px_0_var(--divider-soft)]">
        <tr>
          <th scope="col" className="px-4 py-2 font-medium text-ink-muted">Shared: 2 blocks per stage, {WINDOW}³ windows, MLP ratio 4</th>
          {(['S', 'B', 'L'] as const).map((value) => <th key={value} scope="col" className="px-4 py-2 text-right font-medium">{value}</th>)}
        </tr>
      </thead>
      <tbody>
        {rows.map(([name, value]) => <tr key={name}>
          <th scope="row" className="px-4 py-1.5 font-normal text-ink-muted"><Notation text={name} /></th>
          {(['S', 'B', 'L'] as const).map((column) => <td key={column} className="px-4 py-1.5 text-right">{value(column)}</td>)}
        </tr>)}
      </tbody>
    </table>
  </Panel>;
}

/* ---------- Convolution: one kernel, from a 2D slice to a 3D volume to stride 2 ---------- */

const CONV_EDGE = 8;
const CONV_OFFSET = CONV_EDGE + 3;
const CONV_BOUNDS = new THREE.Box3(new THREE.Vector3(-1.5, -1.5, -1.5), new THREE.Vector3(CONV_OFFSET + CONV_EDGE - 0.5, CONV_EDGE + 0.5, CONV_EDGE + 0.5));
const CONV_STEPS = [{ value: 0, label: '1 · 2D kernel' }, { value: 1, label: '2 · Add depth' }, { value: 2, label: '3 · Stride 2' }];
/** Toy input: a smooth blob, so the edge-detecting kernel has something to respond to. */
const convInput = (x: number, y: number, z: number) => Math.exp(-((x - 3.5) ** 2 + (y - 3.5) ** 2 + (z - 3.5) ** 2) / 6);
const SMOOTH = [1, 2, 1];
/** Toy kernel: change along x, smoothed along y and z (3D Sobel). The 2D step uses its dz = 0 slice. */
const convWeight = (dx: number, dy: number, dz: number) => dx * SMOOTH[dy + 1] * SMOOTH[dz + 1];
const convCoordinates = (index: number) => [Math.floor(index / CONV_EDGE ** 2), Math.floor(index / CONV_EDGE) % CONV_EDGE, index % CONV_EDGE];
const convIndex = ([x, y, z]: number[]) => (x * CONV_EDGE + y) * CONV_EDGE + z;
const inConvGrid = (...values: number[]) => values.every((value) => value >= 0 && value < CONV_EDGE);
const OFFSETS = [-1, 0, 1];

/** Kernel output at `center`, zero padding outside the grid. */
function convOutput(center: number[], depth: boolean): number {
  let sum = 0;
  for (const dx of OFFSETS) for (const dy of OFFSETS) for (const dz of depth ? OFFSETS : [0]) {
    const [x, y, z] = [center[0] + dx, center[1] + dy, center[2] + dz];
    if (inConvGrid(x, y, z)) sum += convWeight(dx, dy, dz) * convInput(x, y, z);
  }
  return sum;
}

/** One 3 × 3 kernel slice: weight on top, the input value it multiplies below, `pad` outside the grid. */
function KernelSlice({ center, dz, label }: { center: number[]; dz: number; label: string }) {
  return <div className="space-y-1">
    <p className="text-[11px] text-ink-muted">{label}</p>
    <div className="grid w-fit grid-cols-3 gap-1 tabular-nums">
      {[1, 0, -1].flatMap((dy) => OFFSETS.map((dx) => {
        const [x, y, z] = [center[0] + dx, center[1] + dy, center[2] + dz];
        const inside = inConvGrid(x, y, z);
        return <div key={`${dx},${dy}`} className="w-11 rounded-md bg-paper px-1 py-0.5 text-center leading-tight">
          <span className="block text-xs font-medium">{convWeight(dx, dy, dz)}</span>
          <span className="block text-[10px] text-ink-muted">{inside ? convInput(x, y, z).toFixed(2) : 'pad'}</span>
        </div>;
      }))}
    </div>
  </div>;
}

/**
 * One toy kernel slid over a toy blob, in three user-driven steps on the same cubes.
 *
 * - 2D: one slice, a 3 × 3 kernel and the whole output feature map
 * - Depth: hidden layers grow out of the slice, the kernel becomes 3 × 3 × 3
 * - Stride 2: odd outputs vanish, the kept ones close up into a 4³ grid
 */
export function ConvolutionScene() {
  const [step, setStep] = useState(0);
  const [center, setCenter] = useState([2, 4, 4]);
  const depth = step > 0;
  const strided = step === 2;
  const stride = strided ? 2 : 1;
  const changeStep = (next: number) => {
    if (next === 2) setCenter(center.map((value) => value - (value % 2)));
    setStep(next);
  };
  const select = (coordinates: number[]) => setCenter(coordinates.map((value, axis) =>
    axis === 2 && !depth ? center[2] : strided ? value - (value % 2) : value));
  const layout = useMemo(() => {
    const voxels = CONV_EDGE ** 3;
    const positions = new Float32Array(2 * voxels * 3);
    const scales = new Float32Array(positions.length);
    const colors = new Float32Array(positions.length);
    const outputs = Array.from({ length: voxels }, (_, index) => convOutput(convCoordinates(index), depth));
    const bound = Math.max(...outputs.map(Math.abs)) || 1;
    const outputScale: ColorScale = { colormap: 'diverging', range: [-bound, bound], log: false };
    for (let index = 0; index < voxels; index++) {
      const [x, y, z] = convCoordinates(index);
      const onSlice = z === center[2];
      const inKernel = Math.abs(x - center[0]) <= 1 && Math.abs(y - center[1]) <= 1 && (depth ? Math.abs(z - center[2]) <= 1 : onSlice);
      // Hidden layers wait on the shown slice, so adding depth grows them out of it. In 2D the kernel lifts toward the viewer
      positions.set([x, y, depth ? z : center[2] + (inKernel ? 0.35 : 0)], index * 3);
      scales.fill(depth ? (inKernel ? 0.86 : 0.26) : onSlice ? (inKernel ? 0.9 : 0.76) : HIDDEN_SCALE, index * 3, index * 3 + 3);
      colors.set(linearColor(colorAt(convInput(x, y, z), TOY_VALUE_SCALE)), index * 3);
      const output = voxels + index;
      const kept = !strided || (x % 2 === 0 && y % 2 === 0 && z % 2 === 0);
      const selected = x === center[0] && y === center[1] && z === center[2];
      positions.set(strided && kept ? [CONV_OFFSET + x / 2 + 2, y / 2 + 2, z / 2 + 2] : [CONV_OFFSET + x, y, depth ? z : center[2] + (selected ? 0.35 : 0)], output * 3);
      scales.fill(!kept || (!depth && !onSlice) ? HIDDEN_SCALE : selected ? 0.9 : !depth ? 0.76 : strided ? 0.5 : 0.26, output * 3, output * 3 + 3);
      colors.set(linearColor(colorAt(outputs[index], outputScale)), output * 3);
    }
    return { positions, scales, colors, value: outputs[convIndex(center)] };
  }, [center, depth, strided]);
  const kernelMin: THREE.Vector3Tuple = [center[0] - 1.5, center[1] - 1.5, center[2] + (depth ? -1.5 : -0.15)];
  const kernelMax: THREE.Vector3Tuple = [center[0] + 1.5, center[1] + 1.5, center[2] + (depth ? 1.5 : 0.85)];
  const outputPoint: THREE.Vector3Tuple = strided ? [CONV_OFFSET + center[0] / 2 + 2, center[1] / 2 + 2, center[2] / 2 + 2] : [CONV_OFFSET + center[0], center[1], center[2] + (depth ? 0 : 0.35)];
  const reads = OFFSETS.flatMap((dx) => OFFSETS.flatMap((dy) => (depth ? OFFSETS : [0]).map((dz) => inConvGrid(center[0] + dx, center[1] + dy, center[2] + dz))));
  const padded = reads.filter((inside) => !inside).length;
  const [inputChannels, outputChannels] = SIZES[SIZE].features;
  const outputEdge = CONV_EDGE / stride;
  const outputIndex = center.map((value) => value / stride);
  const axes = depth ? ['x', 'y', 'z'] : ['x', 'y'];
  return <Panel className="space-y-4 p-4">
    <Segmented label="Convolution step" value={step} onChange={changeStep} options={CONV_STEPS} />
    <p className="max-w-3xl text-sm leading-relaxed">{[
      'The same 9 weights slide over the slice: multiply, add, write one output. This kernel detects change along x, so the two sides of the blob get opposite signs.',
      'In 3D the kernel is 3 × 3 × 3, 27 weights per output.',
      'Stride 2 skips every second position, so each axis halves. Tadpole does this four times, 64³ → 4³.',
    ][step]}</p>
    <div className="grid gap-5 lg:grid-cols-[minmax(0,1.7fr)_minmax(15rem,1fr)]">
      <figure className="min-w-0 space-y-2">
        <Scene bounds={CONV_BOUNDS} height={400} label={`Toy convolution. Left: input ${depth ? `${CONV_EDGE}³ volume` : 'slice'} colored by value, kernel outlined at (${center.join(', ')}). Right: output ${depth ? `${outputEdge}³ volume` : 'feature map'}, the selected output outlined and linked to its kernel.`}>
          <Voxels positions={layout.positions} scales={layout.scales} colors={layout.colors}
            onSelect={(index) => select(convCoordinates(index % CONV_EDGE ** 3))} />
          <Outline min={kernelMin} max={kernelMax} />
          <Outline min={outputPoint.map((value) => value - 0.55) as THREE.Vector3Tuple} max={outputPoint.map((value) => value + 0.55) as THREE.Vector3Tuple} />
          <Segments points={[...center, ...outputPoint]} opacity={0.6} />
        </Scene>
        <figcaption className="text-xs text-ink-muted">Input → output · blue negative, red positive</figcaption>
        <div className={`grid gap-3 ${depth ? 'sm:grid-cols-3' : 'sm:grid-cols-2'}`}>
          {axes.map((axis, index) => <div key={axis} className="space-y-1.5">
            <p className="text-xs tabular-nums">{axis} {outputIndex[index]}</p>
            <Slider label={`Convolution output ${axis}`} min={0} max={outputEdge - 1} step={1} values={[outputIndex[index]]}
              onChange={([next]) => setCenter(center.map((value, axis) => axis === index ? next * stride : value))} />
          </div>)}
        </div>
      </figure>
      <div className="space-y-3 text-sm">
        <p className="text-xs text-ink-muted">Weight, input below</p>
        <div className="flex flex-wrap gap-3">
          {(depth ? [-1, 0, 1] : [0]).map((dz) => <KernelSlice key={dz} center={center} dz={dz} label={depth ? `Slice z ${dz < 0 ? '− 1' : dz > 0 ? '+ 1' : ''}` : 'Kernel'} />)}
        </div>
        <p className="tabular-nums" aria-live="polite">Σ = <strong>{layout.value.toFixed(2)}</strong></p>
        {padded > 0 && <p className="text-xs text-ink-muted tabular-nums">{padded} inputs outside the grid count as 0</p>}
      </div>
    </div>
    <p className="max-w-3xl border-t border-divider-soft pt-3 text-sm leading-relaxed tabular-nums">
      <strong>Real layers have many channels.</strong> Each output channel has its own kernel spanning all input channels. Tadpole-{SIZE}’s first stride-2 layer maps {inputChannels} → {outputChannels} channels, {formatCount(27 * inputChannels)} inputs per output value.
    </p>
  </Panel>;
}

/* ---------- Window attention ---------- */

const ATTENTION_LEVELS = [16, 8, 4];
const ATTENTION_EDGE = 16;
const attentionIndex = ([x, y, z]: readonly number[]) => (x * ATTENTION_EDGE + y) * ATTENTION_EDGE + z;
const ATTENTION_COORDINATES: THREE.Vector3Tuple[] = Array.from({ length: ATTENTION_EDGE ** 3 }, (_, index) =>
  [Math.floor(index / ATTENTION_EDGE ** 2), Math.floor(index / ATTENTION_EDGE) % ATTENTION_EDGE, index % ATTENTION_EDGE]);

/** The same query and window expand from a planar teaching example into spatial attention. */
export function AttentionScene() {
  const { theme } = useTheme();
  const palette = themeColors(theme);
  const [step, setStep] = useState(0);
  const [edge, setEdge] = useState(8);
  const [query, setQuery] = useState<THREE.Vector3Tuple>([0, 3, 0]);
  const [content, setContent] = useState(0);
  const [head, setHead] = useState(0);
  const [rank, setRank] = useState(0);
  const [qx, qy, qz] = query;
  const token = attentionIndex(query);
  const start = query.map((value) => Math.floor(value / WINDOW) * WINDOW);
  const [sx, sy, sz] = start;
  const plane = step === 0;
  const wholeGrid = step === 2;
  const width = SIZES[SIZE].hidden * 2 ** ATTENTION_LEVELS.indexOf(edge);
  const data = useMemo(() => {
    const baseAngle = ([x, y, z]: readonly number[]) => x * 0.83 + y * 1.37 + z * 0.59;
    // Unit two-component features, distinct query/key rotations and a toy position bias.
    const queryAngle = baseAngle([qx, qy, qz]) + content * 1.4 + 0.65 + head * 0.45;
    const q = [Math.cos(queryAngle), Math.sin(queryAngle)];
    const partners = [];
    for (let x = sx; x < sx + WINDOW; x++) for (let y = sy; y < sy + WINDOW; y++)
      for (let z = sz; z < sz + WINDOW; z++) {
        if (plane && z !== qz) continue;
        const coordinate: THREE.Vector3Tuple = [x, y, z];
        const index = attentionIndex(coordinate);
        const angle = baseAngle(coordinate) + (index === token ? content * 1.4 : 0);
        const keyAngle = angle - head * 0.3;
        const cosine = q[0] * Math.cos(keyAngle) + q[1] * Math.sin(keyAngle);
        const bias = 0.12 * Math.cos((x - qx) - 0.7 * (y - qy) + 0.5 * (z - qz));
        partners.push({ index, coordinate, score: Math.exp(2 * cosine + bias), value: Math.sin(angle + 0.4) });
      }
    const total = partners.reduce((sum, partner) => sum + partner.score, 0);
    const weighted = partners.map((partner) => ({ ...partner, weight: partner.score / total }));
    const strongest = [...weighted].sort((a, b) => b.weight - a.weight).slice(0, 4);
    return { partners: weighted, strongest, result: weighted.reduce((sum, partner) => sum + partner.weight * partner.value, 0) };
  }, [qx, qy, qz, sx, sy, sz, token, plane, content, head]);
  const layout = useMemo(() => {
    const positions = new Float32Array(ATTENTION_COORDINATES.length * 3);
    const scales = new Float32Array(positions.length);
    ATTENTION_COORDINATES.forEach(([x, y, z], index) => {
      const inWindow = x >= sx && x < sx + WINDOW && y >= sy && y < sy + WINDOW && z >= sz && z < sz + WINDOW;
      const visible = x < edge && y < edge && z < edge && (wholeGrid || inWindow) && (!plane || z === qz);
      // Hidden layers start on the query plane and unfold without replacing instances.
      positions.set([x - 1.5, y - 1.5, (plane ? qz : z) - 1.5], index * 3);
      scales.fill(!visible ? HIDDEN_SCALE : index === token ? 0.62 : inWindow ? 0.38 : 0.12, index * 3, index * 3 + 3);
    });
    return { positions, scales };
  }, [edge, wholeGrid, plane, qz, sx, sy, sz, token]);
  const colors = useMemo(() => {
    const colors = new Float32Array(ATTENTION_COORDINATES.length * 3);
    const weights = new Map(data.partners.map((partner) => [partner.index, partner.weight]));
    const surface = oklchToLinear(palette.surface);
    const ink = oklchToLinear(palette.ink);
    const featureColor = oklchToLinear(palette.axes.feature);
    const outsideColor = dim(ink, 0.82, surface);
    const maxWeight = data.strongest[0].weight;
    for (let index = 0; index < ATTENTION_COORDINATES.length; index++) {
      const weight = weights.get(index);
      colors.set(index === token ? ink : weight !== undefined ? dim(featureColor, 0.65 * (1 - weight / maxWeight), surface) : outsideColor, index * 3);
    }
    return colors;
  }, [data, token, palette]);
  // Keep the camera fixed while the plane unfolds, weights change or the query moves.
  const bounds = useMemo(() => {
    const min = wholeGrid ? [-2, -2, -2] : [sx - 2, sy - 2, sz - 2];
    const max = wholeGrid ? [edge - 1, edge - 1, edge - 1] : [sx + 3, sy + 3, sz + 3];
    return new THREE.Box3(new THREE.Vector3(...min), new THREE.Vector3(...max));
  }, [wholeGrid, edge, sx, sy, sz]);
  const windows = useMemo(() => {
    const origins: THREE.Vector3Tuple[] = [];
    if (!wholeGrid) return [[sx, sy, sz] as THREE.Vector3Tuple];
    for (let x = 0; x < edge; x += WINDOW) for (let y = 0; y < edge; y += WINDOW)
      for (let z = 0; z < edge; z += WINDOW) origins.push([x, y, z]);
    return origins;
  }, [wholeGrid, edge, sx, sy, sz]);
  const point = (coordinate: readonly number[]) => coordinate.map((value) => value - 1.5) as THREE.Vector3Tuple;
  const selectedPoint = point(data.strongest[rank].coordinate);
  const queryPoint = point(query);
  const selectToken = (index: number) => {
    const coordinate = ATTENTION_COORDINATES[index];
    if (coordinate.every((value) => value < edge) && layout.scales[index * 3] > HIDDEN_SCALE) setQuery(coordinate);
  };
  return <Panel className="space-y-4 p-4">
    <Segmented label="Attention illustration" value={step} onChange={setStep} options={[
      { value: 0, label: '1 · A 2D window' }, { value: 1, label: '2 · Add depth' }, { value: 2, label: '3 · Windows in a grid' },
    ]} />
    <p className="text-sm">{plane
      ? 'The query compares itself with all 16 points of its window, then takes a weighted mix of their values.'
      : wholeGrid ? `Fixed 4³ windows tile the ${edge}³ grid. Points only mix within their own window.`
        : 'The full 4 × 4 × 4 window: 64 partners.'}</p>
    {wholeGrid && <div className="space-y-2">
      <Segmented label="Attention stage grid" value={edge} onChange={(next) => {
        setEdge(next);
        setQuery(query.map((value) => Math.min(value, next - 1)) as THREE.Vector3Tuple);
      }} options={ATTENTION_LEVELS.map((value) => ({ value, label: `${value}³ grid` }))} />
      <p className="text-xs text-ink-muted">{edge === 4 ? 'At 4³ one window covers the whole crop.' : 'Coarser grid, same 64 partners, wider reach.'}</p>
    </div>}
    <div className="grid gap-5 lg:grid-cols-[minmax(0,1.6fr)_minmax(16rem,1fr)]">
      <figure className="min-w-0 space-y-2">
        <Scene bounds={bounds} height={420} label={`Query (${query.join(', ')}) is ${theme === 'dark' ? 'light' : 'dark'}. ${data.partners.length} partners including itself are highlighted. Four strongest illustrative links are shown. ${wholeGrid ? 'Other windows are dim and do not contribute.' : plane ? 'A planar teaching example.' : 'The same window expanded in depth.'}`}>
          <Voxels {...layout} colors={colors} onSelect={selectToken} />
          {windows.map(([x, y, z]) => <Outline key={`${x},${y},${z}`}
            min={[x - 1.95, y - 1.95, (plane ? qz : z) - 1.95]}
            max={[x + 1.95, y + 1.95, (plane ? qz - 1.05 : z + 1.95)]} />)}
          <Outline min={queryPoint.map((value) => value - 0.4) as THREE.Vector3Tuple} max={queryPoint.map((value) => value + 0.4) as THREE.Vector3Tuple} />
          <Outline min={selectedPoint.map((value) => value - 0.27) as THREE.Vector3Tuple} max={selectedPoint.map((value) => value + 0.27) as THREE.Vector3Tuple} />
          {data.strongest.map((partner, index) => <Segments key={partner.index}
            points={[...queryPoint, ...point(partner.coordinate)]} opacity={index === rank ? 1 : 0.25 + 0.4 * partner.weight / data.strongest[0].weight} />)}
        </Scene>
        <figcaption className="text-xs text-ink-muted">{theme === 'dark' ? 'Light' : 'Dark'} = query · stronger color = more weight · lines = strongest four</figcaption>
        <div className="grid grid-cols-3 gap-3">
          {(['x', 'y', 'z'] as const).map((axis, index) => <div key={axis} className="space-y-1.5">
            <p className="text-xs tabular-nums">{axis} {query[index]}</p>
            <Slider label={`Attention query ${axis}`} min={wholeGrid ? 0 : start[index]} max={wholeGrid ? edge - 1 : start[index] + WINDOW - 1} step={1} values={[query[index]]}
              onChange={([value]) => setQuery(query.map((current, coordinate) => coordinate === index ? value : current) as THREE.Vector3Tuple)} />
          </div>)}
        </div>
      </figure>
      <div className="space-y-4 text-sm">
        <section className="space-y-2">
          <h3 className="font-medium">1. Query, keys, values</h3>
          <p className="text-xs text-ink-muted">Learned projections of each point’s features.</p>
          <Segmented label="Query content" value={content} onChange={setContent} options={[{ value: 0, label: 'Original content' }, { value: 1, label: 'Changed content' }]} />
        </section>
        <section className="space-y-2">
          <h3 className="font-medium">2. Compare Q with every K</h3>
          <p className="text-xs text-ink-muted">Similarity, then softmax: weights that sum to 1.</p>
          <div className="grid grid-cols-2 gap-2">
            {data.strongest.map((partner, index) => <button key={index} type="button" aria-pressed={rank === index} onClick={() => setRank(index)}
              className={`rounded-lg px-3 py-2 text-left text-xs tabular-nums focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ink ${rank === index ? 'bg-ink text-paper' : 'bg-paper text-ink hover:bg-hover'}`}>
              <span className="block">#{index + 1} · ({partner.coordinate.join(', ')}){partner.index === token ? ' · self' : ''}</span>
              <span>{(100 * partner.weight).toFixed(1)}% weight</span>
            </button>)}
          </div>
        </section>
        <section className="space-y-2">
          <h3 className="font-medium">3. Sum weighted values</h3>
          <p className="text-xs tabular-nums" aria-live="polite">Σ weight × value = <strong>{data.result.toFixed(3)}</strong></p>
          <p className="text-xs text-ink-muted">The new features of the query.</p>
        </section>
      </div>
    </div>
    <Disclosure label="Heads" className="space-y-3">
      <p>Tadpole-{SIZE} splits the {width} features per point over {SIZES[SIZE].heads} heads. Each head compares the same partners on its own slice of the features, so it can attend to different things.</p>
      <div className="max-w-xs space-y-1.5">
        <p>Head {head + 1}</p>
        <Slider label="Attention head" min={0} max={SIZES[SIZE].heads - 1} step={1} values={[head]} onChange={([next]) => setHead(next)} />
      </div>
    </Disclosure>
  </Panel>;
}

/* ---------- CoolData latent: 400 × 80 × 16 voxels under 25 × 5 × 1 latent cells ---------- */

const COARSE = 4;
const CELL = DOWNSAMPLING / COARSE;
const COOLDATA_STAGES = ['Voxels', 'Height collapses', 'Latent cells'] as const;

/**
 * One field drawn 4× coarser per axis with Tadpole's latent cells over it.
 *
 * - Hovering a voxel or a pillar highlights the cell and its 16³ input block
 * - The stages squeeze the 16 voxel layers into one cell height, then merge each block into its 512-feature pillar
 */
export function CoolDataLatent({ config, field }: { config: Config; field: Field }) {
  const { theme } = useTheme();
  const palette = themeColors(theme);
  const [nx, ny, nz] = config.dimensions;
  const [cx, cy, cz] = [nx, ny, nz].map((voxels) => voxels / COARSE);
  const [stage, setStage] = useState(0);
  const [cell, setCell] = useState<[number, number] | undefined>(undefined);
  const coarse = useMemo(() => {
    const values = new Float32Array(cx * cy * cz);
    const solid = new Uint8Array(cx * cy * cz);
    for (let x = 0; x < cx; x++) for (let y = 0; y < cy; y++) for (let z = 0; z < cz; z++) {
      let sum = 0;
      let fluid = 0;
      for (let dx = 0; dx < COARSE; dx++) for (let dy = 0; dy < COARSE; dy++) for (let dz = 0; dz < COARSE; dz++) {
        const index = x * COARSE + dx + nx * (y * COARSE + dy + ny * (z * COARSE + dz));
        if (field.mask[index]) { sum += field.values[index]; fluid++; }
      }
      const coarseIndex = (x * cy + y) * cz + z;
      values[coarseIndex] = fluid ? sum / fluid : 0;
      solid[coarseIndex] = fluid * 2 < COARSE ** 3 ? 1 : 0;
    }
    return { values, solid };
  }, [field, nx, ny, cx, cy, cz]);
  const pillar = pillarHeight(512);
  const layout = useMemo(() => {
    const count = cx * cy * cz;
    const positions = new Float32Array(count * 3);
    const scales = new Float32Array(count * 3);
    for (let index = 0; index < count; index++) {
      const [x, y, z] = [Math.floor(index / (cy * cz)), Math.floor(index / cz) % cy, index % cz];
      const cellCenter = [Math.floor(x / CELL) * CELL + (CELL - 1) / 2, Math.floor(y / CELL) * CELL + (CELL - 1) / 2];
      if (stage === 0) { positions.set([x, z, cy - 1 - y], index * 3); scales.set([0.55, 0.55, 0.55], index * 3); continue; }
      if (stage === 1) { positions.set([x, (cz - 1) / 2 + (z - (cz - 1) / 2) * 0.12, cy - 1 - y], index * 3); scales.set([0.9, 0.11, 0.9], index * 3); continue; }
      const isRepresentative = x % CELL === 0 && y % CELL === 0 && z === 0;
      positions.set([cellCenter[0], isRepresentative ? pillar / 2 : (cz - 1) / 2, cy - 1 - cellCenter[1]], index * 3);
      scales.set(isRepresentative ? [CELL * 0.8, pillar, CELL * 0.8] : [HIDDEN_SCALE, HIDDEN_SCALE, HIDDEN_SCALE], index * 3);
    }
    return { positions, scales };
  }, [stage, cx, cy, cz, pillar]);
  const colors = useMemo(() => {
    const colors = new Float32Array(cx * cy * cz * 3);
    const surface = oklchToLinear(palette.surface);
    const solid = oklchToLinear(palette.solid);
    const feature = oklchToLinear(palette.axes.feature);
    const neutral = featureGray(palette, 0.22);
    const temperatureScale: ColorScale = { colormap: 'sequential', range: [field.min, field.max], log: false };
    for (let index = 0; index < cx * cy * cz; index++) {
      const [x, y] = [Math.floor(index / (cy * cz)), Math.floor(index / cz) % cy];
      const inCell = cell !== undefined && Math.floor(x / CELL) === cell[0] && Math.floor(y / CELL) === cell[1];
      const base = stage === 2 ? neutral : coarse.solid[index] ? solid
        : linearColor(colorAt(coarse.values[index], temperatureScale));
      colors.set(stage === 2 && inCell ? feature : dim(base, cell && !inCell ? 0.55 : 0, surface), index * 3);
    }
    return colors;
  }, [coarse, cell, stage, field, cx, cy, cz, palette]);
  const bounds = useMemo(() => {
    const box = boundsOf([layout]);
    return box.expandByPoint(new THREE.Vector3(0, pillar, 0));
  }, [layout, pillar]);
  const hover = (index: number | undefined) => {
    if (index === undefined) return setCell(undefined);
    setCell([Math.floor(Math.floor(index / (cy * cz)) / CELL), Math.floor((Math.floor(index / cz) % cy) / CELL)]);
  };
  const [cellsX, cellsY] = [nx, ny].map((voxels) => voxels / DOWNSAMPLING);
  const top = stage === 0 ? cz - 0.5 : stage === 1 ? (cz - 1) / 2 + 0.06 * cz : pillar;
  const bottom = stage === 1 ? (cz - 1) / 2 - 0.06 * cz : -0.5;
  const cellGrid = useMemo(() => {
    const points: number[] = [];
    const [front, back, right] = [cy - 0.5, -0.5, cx - 0.5];
    for (let column = 0; column <= cellsX; column++) {
      const x = column * CELL - 0.5;
      points.push(x, top, back, x, top, front, x, top, front, x, bottom, front);
    }
    for (let row = 0; row <= cellsY; row++) {
      const z = front - row * CELL;
      points.push(-0.5, top, z, right, top, z, right, top, z, right, bottom, z);
    }
    return points;
  }, [cellsX, cellsY, cx, cy, top, bottom]);
  const outline = cell && <Outline min={[cell[0] * CELL - 0.5, bottom, cy - (cell[1] + 1) * CELL - 0.5]} max={[(cell[0] + 1) * CELL - 0.5, top + (stage === 2 ? 0.1 : 0), cy - cell[1] * CELL - 0.5]} />;
  return <Panel className="space-y-4 p-4">
    <Scene bounds={bounds} height={340} label={`Field on the ${nx} by ${ny} by ${nz} grid with Tadpole's ${cellsX} by ${cellsY} by 1 latent cells`}>
      <Voxels {...layout} colors={colors} onHover={hover} />
      {stage < 2 && <Segments points={cellGrid} opacity={0.7} />}
      {outline}
    </Scene>
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between gap-3 text-xs tabular-nums">
        <span>{cell
          ? <>Latent cell ({cell[0]}, {cell[1]}, 0) ← voxels x {cell[0] * DOWNSAMPLING}–{cell[0] * DOWNSAMPLING + DOWNSAMPLING - 1}, y {cell[1] * DOWNSAMPLING}–{cell[1] * DOWNSAMPLING + DOWNSAMPLING - 1}, z 0–{nz - 1}: {formatCount(DOWNSAMPLING ** 3)} voxels → <span style={{ color: oklchCss(palette.axes.feature) }}>512</span> features</>
          : <span className="text-ink-muted">Hover a voxel or a pillar for its latent cell. Drawn {COARSE}× coarser per axis, {formatCount(cx * cy * cz)} of {formatCount(nx * ny * nz)} voxels</span>}</span>
        <span className="shrink-0 text-ink-muted">{COOLDATA_STAGES[stage]}</span>
      </div>
      <Slider label="Stage" min={0} max={COOLDATA_STAGES.length - 1} step={1} values={[stage]} onChange={([next]) => setStage(next)} />
    </div>
  </Panel>;
}


