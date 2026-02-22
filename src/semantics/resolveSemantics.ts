import {
  applyOpacityPolicy,
  applySigmaPolicy,
  decodeColorDC,
  sanitizeFiniteInPlace,
  toFloat32Array,
} from '../utils/tensorUtils';
import type { ParsedModel, SemanticData, TensorRecord } from '../types';

const KEY_ALIASES = {
  positions: ['triangles_points', 'triangle_points', 'positions'],
  colorDC: ['features_dc', 'color_dc', 'colors_dc'],
  colorRest: ['features_rest', 'color_rest', 'colors_rest'],
  opacity: ['opacity_alpha', 'opacity', 'alpha'],
  sigma: ['sigma'],
} as const;

interface ResolveOptions {
  opacityMode?: 'auto' | 'linear' | 'logits';
  colorMode?: 'auto' | 'linear' | 'sh';
  sigmaMode?: 'auto' | 'linear' | 'log';
}

function findTensor(
  tensors: Record<string, TensorRecord>,
  aliases: readonly string[],
): TensorRecord | null {
  for (const key of aliases) {
    if (tensors[key]) {
      return tensors[key];
    }
  }
  return null;
}

function findTensorName(
  tensors: Record<string, TensorRecord>,
  aliases: readonly string[],
): string | undefined {
  return aliases.find((k) => Boolean(tensors[k]));
}

export function resolveSemantics(
  parsedModel: ParsedModel,
  options: ResolveOptions = {},
): SemanticData {
  const tensors = parsedModel?.tensors ?? {};
  const opacityMode = options.opacityMode ?? 'auto';
  const colorMode = options.colorMode ?? 'auto';
  const sigmaMode = options.sigmaMode ?? 'auto';

  const positionTensor = findTensor(tensors, KEY_ALIASES.positions);
  if (!positionTensor) {
    throw new Error(
      `Missing position tensor. Expected one of: ${KEY_ALIASES.positions.join(', ')}`,
    );
  }

  const positionData = toFloat32Array(positionTensor.data);
  const replacedPositionValues = sanitizeFiniteInPlace(positionData);

  if (!Array.isArray(positionTensor.shape) || positionTensor.shape.length < 3) {
    throw new Error(
      `Invalid position tensor shape '${JSON.stringify(positionTensor.shape)}'. Expected [N,3,3]`,
    );
  }

  const triangleCount = Number(positionTensor.shape[0]);
  const verticesPerTriangle = Number(positionTensor.shape[1]);
  const coordsPerVertex = Number(positionTensor.shape[2]);

  if (verticesPerTriangle !== 3 || coordsPerVertex !== 3) {
    throw new Error(
      `Unsupported position shape [${triangleCount},${verticesPerTriangle},${coordsPerVertex}]. Expected [N,3,3]`,
    );
  }

  const totalVertices = triangleCount * verticesPerTriangle;
  const colorTensor = findTensor(tensors, KEY_ALIASES.colorDC);

  const colors = new Float32Array(totalVertices * 3).fill(0.5);
  let colorInterpretation = 'none';

  if (colorTensor) {
    const src = toFloat32Array(colorTensor.data);
    const decoded = decodeColorDC(src, colorTensor.dtype ?? 'f32', colorMode);
    colorInterpretation = decoded.interpretation;

    for (let tri = 0; tri < triangleCount; tri++) {
      const base = tri * 3;
      const r = decoded.values[base] ?? 0.5;
      const g = decoded.values[base + 1] ?? 0.5;
      const b = decoded.values[base + 2] ?? 0.5;
      for (let v = 0; v < 3; v++) {
        const dst = (tri * 3 + v) * 3;
        colors[dst] = r;
        colors[dst + 1] = g;
        colors[dst + 2] = b;
      }
    }
  }

  const opacityTensor = findTensor(tensors, KEY_ALIASES.opacity);
  let alphas: Float32Array | null = null;
  let opacityInterpretation = 'none';

  if (opacityTensor) {
    const src = toFloat32Array(opacityTensor.data);
    const applied = applyOpacityPolicy(src, opacityMode);
    alphas = new Float32Array(totalVertices);
    for (let tri = 0; tri < triangleCount; tri++) {
      const alpha = applied.values[tri] ?? 1;
      for (let v = 0; v < 3; v++) {
        alphas[tri * 3 + v] = alpha;
      }
    }
    opacityInterpretation = applied.treatedAs;
  }

  const sigmaTensor = findTensor(tensors, KEY_ALIASES.sigma);
  let sigmaValues: Float32Array | null = null;
  let sigmaInterpretation = 'none';

  if (sigmaTensor) {
    const src = toFloat32Array(sigmaTensor.data);
    const applied = applySigmaPolicy(src, sigmaMode);
    sigmaValues = new Float32Array(totalVertices);
    for (let tri = 0; tri < triangleCount; tri++) {
      const sigma = applied.values[tri] ?? 0.1;
      for (let v = 0; v < 3; v++) {
        sigmaValues[tri * 3 + v] = sigma;
      }
    }
    sigmaInterpretation = applied.treatedAs;
  }

  const shDegree = Number(
    parsedModel?.header?.params?.active_sh_degree ?? parsedModel?.header?.active_sh_degree ?? 0,
  );

  return {
    positions: {
      tensorName: findTensorName(tensors, KEY_ALIASES.positions),
      data: positionData,
      shape: positionTensor.shape,
    },
    colorDC: colorTensor
      ? {
          tensorName: findTensorName(tensors, KEY_ALIASES.colorDC),
          data: colors,
          interpretation: colorInterpretation,
        }
      : null,
    colorRest: findTensor(tensors, KEY_ALIASES.colorRest),
    opacity: alphas ? { data: alphas, interpretation: opacityInterpretation } : null,
    sigma: sigmaValues ? { data: sigmaValues, interpretation: sigmaInterpretation } : null,
    shDegree,
    stats: {
      triangleCount,
      totalVertices,
      replacedPositionValues,
      colorInterpretation,
      opacityInterpretation,
      sigmaInterpretation,
    },
  };
}
