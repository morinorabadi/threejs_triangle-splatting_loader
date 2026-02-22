import { detectFormat } from './formats/detectFormat';
import { parseTGSP } from './formats/tgspParser';
import { parseTSGD } from './formats/tsgdParser';
import type { ParsedModel } from './types';

export class FormatLoader {
  static async loadFile(file: File): Promise<ParsedModel> {
    const buffer = await file.arrayBuffer();
    return this.parse(buffer);
  }

  static async loadURL(url: string): Promise<ParsedModel> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to load URL '${url}' (${response.status})`);
    }
    const buffer = await response.arrayBuffer();
    return this.parse(buffer);
  }

  static parse(buffer: ArrayBuffer): ParsedModel {
    const detected = detectFormat(buffer);

    if (detected.parser === 'tgsp') {
      return parseTGSP(buffer);
    }

    if (detected.parser === 'tsgd') {
      return parseTSGD(buffer, detected.magic as 'TSGD' | 'TGSD');
    }

    throw new Error(`No parser available for '${detected.format}'`);
  }
}
