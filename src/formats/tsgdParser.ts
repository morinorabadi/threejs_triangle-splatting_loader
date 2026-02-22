import { parseContainer } from './containerParser';
import type { ParsedModel } from '../types';

// Adapter scaffold: currently uses TGSP-style container layout with a TSGD magic.
export function parseTSGD(buffer: ArrayBuffer, magic: 'TSGD' | 'TGSD' = 'TSGD'): ParsedModel {
  return parseContainer(buffer, magic);
}
