declare module 'node:crypto' {
  export interface Hash {
    update(data: Uint8Array): Hash;
    digest(encoding: 'hex'): string;
  }
  export function createHash(algorithm: string): Hash;
}

declare module 'node:fs' {
  export function readFileSync(path: string): Uint8Array;
  export function writeFileSync(path: string, data: Uint8Array): void;
  export function existsSync(path: string): boolean;
  export function mkdirSync(path: string, options?: { recursive?: boolean }): void;
  export function statSync(path: string): { size: number };
}
