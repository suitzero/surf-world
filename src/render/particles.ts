export const ParticleKind = {
  WAKE: 0,
  SPRAY: 1,
  STREAK: 2,
} as const;

export type ParticleKind = (typeof ParticleKind)[keyof typeof ParticleKind];

export class ParticleSystem {
  public readonly capacity: number;
  public readonly x: Float32Array;
  public readonly y: Float32Array;
  public readonly vx: Float32Array;
  public readonly vy: Float32Array;
  public readonly life: Float32Array;
  public readonly maxLife: Float32Array;
  public readonly size: Float32Array;
  public readonly kind: Uint8Array;

  private cursor = 0;
  private rngState: number;

  constructor(capacity = 720, seed = 0x9e3779b9) {
    this.capacity = capacity;
    this.x = new Float32Array(capacity);
    this.y = new Float32Array(capacity);
    this.vx = new Float32Array(capacity);
    this.vy = new Float32Array(capacity);
    this.life = new Float32Array(capacity);
    this.maxLife = new Float32Array(capacity);
    this.size = new Float32Array(capacity);
    this.kind = new Uint8Array(capacity);
    this.rngState = seed >>> 0;
  }

  private nextRand(): number {
    let t = (this.rngState = (this.rngState + 0x6d2b79f5) | 0);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  private alloc(): number {
    const idx = this.cursor;
    this.cursor = (this.cursor + 1) % this.capacity;
    return idx;
  }

  emitWake(px: number, py: number, dirX: number, dirY: number, speed: number): void {
    const idx = this.alloc();
    const perpX = -dirY;
    const perpY = dirX;
    const perpOffset = this.nextRand() - 0.5;
    const perpJitter = (this.nextRand() - 0.5) * 0.8;
    const ml = 0.7 + this.nextRand() * 0.4;
    this.x[idx] = px + perpX * perpOffset;
    this.y[idx] = py + perpY * perpOffset;
    this.vx[idx] = -dirX * speed * 0.15 + perpX * perpJitter;
    this.vy[idx] = -dirY * speed * 0.15 + perpY * perpJitter;
    this.maxLife[idx] = ml;
    this.life[idx] = ml;
    this.size[idx] = 0.5 + this.nextRand() * 0.4;
    this.kind[idx] = ParticleKind.WAKE;
  }

  emitSpray(px: number, py: number, dirX: number, dirY: number, _speed: number): void {
    const idx = this.alloc();
    const perpX = -dirY;
    const perpY = dirX;
    const back = 0.8 + this.nextRand() * 1.4;
    const side = (this.nextRand() - 0.5) * 3;
    const ml = 0.3 + this.nextRand() * 0.2;
    this.x[idx] = px;
    this.y[idx] = py;
    this.vx[idx] = -dirX * back + perpX * side;
    this.vy[idx] = -dirY * back + perpY * side;
    this.maxLife[idx] = ml;
    this.life[idx] = ml;
    this.size[idx] = 0.10 + this.nextRand() * 0.12;
    this.kind[idx] = ParticleKind.SPRAY;
  }

  emitStreak(px: number, py: number, pvx: number, pvy: number): void {
    const idx = this.alloc();
    const ml = 0.25 + this.nextRand() * 0.15;
    this.x[idx] = px;
    this.y[idx] = py;
    this.vx[idx] = pvx;
    this.vy[idx] = pvy;
    this.maxLife[idx] = ml;
    this.life[idx] = ml;
    this.size[idx] = 0.12;
    this.kind[idx] = ParticleKind.STREAK;
  }

  update(dt: number): void {
    const drag = Math.max(0, 1 - 2.2 * dt);
    const n = this.capacity;
    for (let i = 0; i < n; i++) {
      if (this.life[i] <= 0) continue;
      this.life[i] -= dt;
      this.x[i] += this.vx[i] * dt;
      this.y[i] += this.vy[i] * dt;
      if (this.kind[i] === ParticleKind.WAKE) {
        this.vx[i] *= drag;
        this.vy[i] *= drag;
      }
    }
  }

  get activeCount(): number {
    let count = 0;
    for (let i = 0; i < this.capacity; i++) {
      if (this.life[i] > 0) count++;
    }
    return count;
  }
}
