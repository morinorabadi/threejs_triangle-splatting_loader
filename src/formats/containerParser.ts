import type { ParsedModel, TensorDType, TensorRecord } from '../types';

const FILE_HEADER_SIZE = 32;
const TOC_ENTRY_SIZE = 32;

const DTYPE_CODE_TO_NAME: Record<number, TensorDType> = {
  1: 'f16',
  2: 'f32',
  3: 'u8',
  4: 'i32',
  5: 'i64',
};

const DTYPE_BYTES: Record<Exclude<TensorDType, 'unknown'>, number> = {
  f16: 2,
  f32: 4,
  u8: 1,
  i32: 4,
  i64: 8,
};

function getMagic(buffer: ArrayBuffer): string {
  return String.fromCharCode(...new Uint8Array(buffer, 0, 4));
}

function product(shape: number[]): number {
  if (!Array.isArray(shape) || shape.length === 0) {
    return 0;
  }

  return shape.reduce((acc, dim) => {
    const n = Number(dim);
    if (!Number.isInteger(n) || n < 0) {
      throw new Error(`Invalid tensor shape dimension: ${dim}`);
    }
    return acc * n;
  }, 1);
}

function float16ToFloat32(fp16: number): number {
  const sign = (fp16 & 0x8000) >> 15;
  const exponent = (fp16 & 0x7c00) >> 10;
  const fraction = fp16 & 0x03ff;

  if (exponent === 0) {
    if (fraction === 0) {
      return 0;
    }
    return (sign ? -1 : 1) * Math.pow(2, -14) * (fraction / Math.pow(2, 10));
  }

  if (exponent === 31) {
    return sign ? -65504 : 65504;
  }

  return (
    (sign ? -1 : 1) * Math.pow(2, exponent - 15) * (1 + fraction / Math.pow(2, 10))
  );
}

function readTypedTensor(
  buffer: ArrayBuffer,
  absoluteOffset: number,
  nbytes: number,
  dtype: Exclude<TensorDType, 'unknown'>,
): TensorRecord['data'] {
  const raw = new Uint8Array(buffer, absoluteOffset, nbytes);
  // JSON header length can make tensor offsets unaligned for TypedArray constructors.
  // Slice to a fresh 0-based buffer to avoid alignment range errors.
  const packed = raw.slice().buffer;

  switch (dtype) {
    case 'f32':
      return new Float32Array(packed);
    case 'f16': {
      const src = new Uint16Array(packed);
      const out = new Float32Array(src.length);
      for (let i = 0; i < src.length; i++) {
        out[i] = float16ToFloat32(src[i]);
      }
      return out;
    }
    case 'u8':
      return new Uint8Array(packed);
    case 'i32':
      return new Int32Array(packed);
    case 'i64':
      return new BigInt64Array(packed);
    default:
      throw new Error(`Unsupported dtype '${dtype}'`);
  }
}

export function parseContainer(
  buffer: ArrayBuffer,
  expectedFormat: 'TGSP' | 'TSGD' | 'TGSD',
): ParsedModel {
  if (!(buffer instanceof ArrayBuffer)) {
    throw new Error('Expected ArrayBuffer input');
  }

  if (buffer.byteLength < FILE_HEADER_SIZE) {
    throw new Error(
      `File too small. Expected at least ${FILE_HEADER_SIZE} bytes, got ${buffer.byteLength}`,
    );
  }

  const view = new DataView(buffer);
  const magic = getMagic(buffer);

  if (magic !== expectedFormat) {
    throw new Error(`Invalid ${expectedFormat} file: bad magic '${magic}'`);
  }

  let offset = 4;
  const version = view.getUint32(offset, true);
  offset += 4;
  const headerLen = view.getUint32(offset, true);
  offset += 4;
  const tocLen = view.getUint32(offset, true);
  offset += 4;

  offset += 16; // reserved

  if (tocLen % TOC_ENTRY_SIZE !== 0) {
    throw new Error(`Invalid TOC length ${tocLen}. Must be a multiple of ${TOC_ENTRY_SIZE}`);
  }

  const dataSectionStart = FILE_HEADER_SIZE + headerLen + tocLen;
  if (dataSectionStart > buffer.byteLength) {
    throw new Error(
      `Header + TOC exceed file size (${dataSectionStart} > ${buffer.byteLength})`,
    );
  }

  const headerBytes = new Uint8Array(buffer, FILE_HEADER_SIZE, headerLen);
  let header: ParsedModel['header'];
  try {
    header = JSON.parse(new TextDecoder().decode(headerBytes));
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Invalid JSON header: ${message}`);
  }

  if (!header || !Array.isArray(header.tensors)) {
    throw new Error('Invalid container header: missing tensors array');
  }

  const tensorCount = tocLen / TOC_ENTRY_SIZE;
  if (header.tensors.length < tensorCount) {
    throw new Error(
      `Header tensor count (${header.tensors.length}) is smaller than TOC entries (${tensorCount})`,
    );
  }

  const tensors: ParsedModel['tensors'] = {};
  offset = FILE_HEADER_SIZE + headerLen;

  for (let i = 0; i < tensorCount; i++) {
    // name_hash is present for lookups, but we currently map tensors by header order.
    view.getBigUint64(offset, true);
    offset += 8;

    const relOffset = Number(view.getBigUint64(offset, true));
    offset += 8;

    const nbytes = Number(view.getBigUint64(offset, true));
    offset += 8;

    const dtypeCode = view.getUint32(offset, true);
    offset += 4;
    offset += 4; // reserved

    const dtype = DTYPE_CODE_TO_NAME[dtypeCode];
    if (!dtype || dtype === 'unknown') {
      throw new Error(`Unknown dtype code '${dtypeCode}' at TOC index ${i}`);
    }

    const def: { name?: string; shape?: number[] } = header.tensors[i] ?? {};
    const name = String(def.name ?? `tensor_${i}`);
    const shape = Array.isArray(def.shape) ? def.shape.map(Number) : [];

    const absOffset = dataSectionStart + relOffset;
    if (relOffset < 0 || nbytes < 0 || absOffset + nbytes > buffer.byteLength) {
      throw new Error(
        `Invalid tensor bounds for '${name}' (offset=${relOffset}, nbytes=${nbytes})`,
      );
    }

    const elemCountFromShape = product(shape);
    if (elemCountFromShape > 0) {
      const expectedNbytes = elemCountFromShape * DTYPE_BYTES[dtype];
      if (expectedNbytes !== nbytes) {
        throw new Error(
          `Tensor '${name}' size mismatch. shape*dtype=${expectedNbytes}, toc=${nbytes}`,
        );
      }
    }

    const data = readTypedTensor(buffer, absOffset, nbytes, dtype);
    tensors[name] = { data, shape, dtype };
  }

  return {
    format: expectedFormat === 'TGSP' ? 'TGSP' : 'TSGD',
    version,
    header,
    tensors,
    metadata: {
      format: expectedFormat === 'TGSP' ? 'TGSP' : 'TSGD',
      version,
      tensorCount,
      names: Object.keys(tensors),
      dataSectionStart,
    },
  };
}
