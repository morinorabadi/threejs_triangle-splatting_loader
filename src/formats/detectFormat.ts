export interface DetectedFormat {
  format: 'TGSP' | 'TSGD';
  parser: 'tgsp' | 'tsgd';
  magic: string;
}

export function detectFormat(buffer: ArrayBuffer): DetectedFormat {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 4) {
    throw new Error('Buffer too small to detect file format');
  }

  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));

  if (magic === 'TGSP') {
    return { format: 'TGSP', parser: 'tgsp', magic };
  }

  if (magic === 'TSGD' || magic === 'TGSD') {
    return { format: 'TSGD', parser: 'tsgd', magic };
  }

  throw new Error(`Unsupported file format magic '${magic}'`);
}
