import type { TensorData, TensorDType } from '../types';

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

const SH_C0 = 0.28209479177387814;

type ColorMode = 'auto' | 'linear' | 'sh';
type OpacityMode = 'auto' | 'linear' | 'logits';
type SigmaMode = 'auto' | 'linear' | 'log';

export function toFloat32Array(input: TensorData): Float32Array {
  if (input instanceof Float32Array) {
    return input;
  }

  if (
    input instanceof Float64Array ||
    input instanceof Int32Array ||
    input instanceof Uint8Array ||
    input instanceof Uint16Array ||
    input instanceof Uint8ClampedArray ||
    input instanceof Int8Array ||
    input instanceof Int16Array ||
    input instanceof Uint32Array
  ) {
    return Float32Array.from(input);
  }

  if (input instanceof BigInt64Array || input instanceof BigUint64Array) {
    const out = new Float32Array(input.length);
    for (let i = 0; i < input.length; i++) {
      out[i] = Number(input[i]);
    }
    return out;
  }

  if (ArrayBuffer.isView(input)) {
    return Float32Array.from(input as unknown as ArrayLike<number>);
  }

  throw new Error(`Cannot convert tensor data type '${typeof input}' to Float32Array`);
}

export function sanitizeFiniteInPlace(values: Float32Array): number {
  let replaced = 0;
  for (let i = 0; i < values.length; i++) {
    if (Number.isFinite(values[i])) {
      continue;
    }

    let replacement = 0;
    for (let j = 1; j < 32; j++) {
      if (i - j >= 0 && Number.isFinite(values[i - j])) {
        replacement = values[i - j];
        break;
      }
      if (i + j < values.length && Number.isFinite(values[i + j])) {
        replacement = values[i + j];
        break;
      }
    }

    values[i] = replacement;
    replaced++;
  }
  return replaced;
}

export function decodeColorDC(
  colorValues: Float32Array,
  dtype: TensorDType = 'f32',
  mode: ColorMode = 'auto',
): { values: Float32Array; interpretation: string } {
  const out = new Float32Array(colorValues.length);
  let interpretation = 'linear';

  if (dtype === 'u8') {
    interpretation = 'u8';
    for (let i = 0; i < colorValues.length; i++) {
      out[i] = colorValues[i] / 255;
    }
  } else {
    let treatAsSH = mode === 'sh';
    if (mode === 'auto') {
      for (let i = 0; i < colorValues.length; i++) {
        if (colorValues[i] < 0 || colorValues[i] > 1) {
          treatAsSH = true;
          break;
        }
      }
    }

    interpretation = treatAsSH ? 'sh' : 'linear';
    for (let i = 0; i < colorValues.length; i++) {
      out[i] = treatAsSH ? colorValues[i] * SH_C0 + 0.5 : colorValues[i];
    }
  }

  for (let i = 0; i < out.length; i++) {
    out[i] = Math.max(0, Math.min(1, out[i]));
  }

  return { values: out, interpretation };
}

export function applyOpacityPolicy(
  opacityValues: Float32Array,
  mode: OpacityMode = 'auto',
): { values: Float32Array; treatedAs: 'logits' | 'linear' } {
  const out = new Float32Array(opacityValues.length);

  let treatAsLogits = false;
  if (mode === 'logits') {
    treatAsLogits = true;
  } else if (mode === 'auto') {
    for (let i = 0; i < opacityValues.length; i++) {
      if (opacityValues[i] < 0 || opacityValues[i] > 1) {
        treatAsLogits = true;
        break;
      }
    }
  }

  for (let i = 0; i < opacityValues.length; i++) {
    const v = treatAsLogits ? sigmoid(opacityValues[i]) : opacityValues[i];
    out[i] = Math.max(0, Math.min(1, v));
  }

  return { values: out, treatedAs: treatAsLogits ? 'logits' : 'linear' };
}

export function applySigmaPolicy(
  sigmaValues: Float32Array,
  mode: SigmaMode = 'auto',
): { values: Float32Array; treatedAs: 'log' | 'linear' } {
  const out = new Float32Array(sigmaValues.length);

  let treatAsLogSigma = false;
  if (mode === 'log') {
    treatAsLogSigma = true;
  } else if (mode === 'auto') {
    for (let i = 0; i < sigmaValues.length; i++) {
      if (sigmaValues[i] <= 0) {
        treatAsLogSigma = true;
        break;
      }
    }
  }

  for (let i = 0; i < sigmaValues.length; i++) {
    const v = treatAsLogSigma ? 0.01 + Math.exp(sigmaValues[i]) : sigmaValues[i];
    out[i] = Math.max(1e-4, Math.min(10, v));
  }

  return { values: out, treatedAs: treatAsLogSigma ? 'log' : 'linear' };
}
