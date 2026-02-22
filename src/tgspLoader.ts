/**
 * Legacy TGSP loader.
 * Kept for compatibility; FormatLoader is the preferred entrypoint.
 */

import { parseContainer } from './formats/containerParser';
import type { ParsedModel } from './types';

export class TGSPLoader {
  static async loadFile(file: File): Promise<ParsedModel> {
    const buffer = await file.arrayBuffer();
    return this.parse(buffer);
  }

  static async loadURL(url: string): Promise<ParsedModel> {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to load: ${url}`);
    const buffer = await response.arrayBuffer();
    return this.parse(buffer);
  }

  static parse(buffer: ArrayBuffer): ParsedModel {
    return parseContainer(buffer, 'TGSP');
  }
}
