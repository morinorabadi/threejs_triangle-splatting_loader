import type * as THREE from 'three';

export type TensorDType = 'f16' | 'f32' | 'u8' | 'i32' | 'i64' | 'unknown';

export type TensorData =
  | Float32Array
  | Float64Array
  | Int32Array
  | Uint8Array
  | BigInt64Array
  | Uint16Array
  | Uint8ClampedArray
  | Int8Array
  | Int16Array
  | Uint32Array
  | BigUint64Array;

export interface TensorRecord {
  data: TensorData;
  shape: number[];
  dtype: TensorDType;
}

export interface ParsedModel {
  format: 'TGSP' | 'TSGD';
  version: number;
  header: {
    tensors?: Array<{ name?: string; shape?: number[]; dtype?: string }>;
    params?: { active_sh_degree?: number };
    active_sh_degree?: number;
    [key: string]: unknown;
  };
  tensors: Record<string, TensorRecord>;
  metadata: {
    format: 'TGSP' | 'TSGD';
    version: number;
    tensorCount: number;
    names: string[];
    dataSectionStart: number;
  };
}

export interface SemanticData {
  positions: {
    tensorName?: string;
    data: Float32Array;
    shape: number[];
  };
  colorDC: {
    tensorName?: string;
    data: Float32Array;
    interpretation: string;
  } | null;
  colorRest: TensorRecord | null;
  opacity: {
    data: Float32Array;
    interpretation: string;
  } | null;
  sigma: {
    data: Float32Array;
    interpretation: string;
  } | null;
  shDegree: number;
  stats: {
    triangleCount: number;
    totalVertices: number;
    replacedPositionValues: number;
    colorInterpretation: string;
    opacityInterpretation: string;
    sigmaInterpretation: string;
  };
}

export interface SplatParams {
  alphaCurveK: number;
  sigmaScale: number;
  coveredAlphaFloor: number;
}

export interface Renderable {
  object3d: THREE.Object3D;
  mode: 'mesh' | 'splat';
  stats: {
    triangles: number;
    points: number;
  };
  render?: (
    renderer: THREE.WebGLRenderer,
    scene: THREE.Scene,
    camera: THREE.Camera,
  ) => void;
  dispose?: () => void;
  setSplatParams?: (params: Partial<SplatParams>) => void;
}
