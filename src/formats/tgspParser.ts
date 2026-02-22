import { parseContainer } from './containerParser';
import type { ParsedModel } from '../types';

export function parseTGSP(buffer: ArrayBuffer): ParsedModel {
  return parseContainer(buffer, 'TGSP');
}
