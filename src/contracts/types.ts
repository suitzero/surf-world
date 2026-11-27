export const CONTRACT_VERSION = 1 as const;

/** Grid geometry shared by bathymetry, ocean field and renderer. */
export const GRID_W = 128;
export const GRID_H = 96;
export const CELL_M = 4; // metres per cell -> world is 512 m x 384 m
export const WORLD_W_M = GRID_W * CELL_M;
export const WORLD_H_M = GRID_H * CELL_M;

export type BathymetryKind = 'beach' | 'sandbar' | 'point' | 'reef';

export const BATHYMETRY_KINDS: readonly BathymetryKind[] = [
  'beach',
  'sandbar',
  'point',
  'reef',
];

/** Offshore swell description (the only wave input to the ocean solver). */
export interface SwellParams {
  /** Offshore wave height H0, metres. */
  heightM: number;
  /** Peak period T, seconds. */
  periodS: number;
  /**
   * Direction the swell travels, degrees from straight-onshore (+y).
   * Positive angles tilt travel toward +x (east).
   */
  directionDeg: number;
}

export interface WorldConfig {
  version: typeof CONTRACT_VERSION;
  gridW: number;
  gridH: number;
  cellM: number;
  /** Fixed simulation step, seconds. */
  dt: number;
  /** Tide offset added to every depth, metres. */
  tideM: number;
  bathymetry: BathymetryKind;
  /** Number of AI surfers (player is extra). 0..8 */
  aiCount: number;
}

/** Seafloor: depth in metres per cell, positive = water, <= 0 = land. */
export interface Bathymetry {
  kind: BathymetryKind;
  seed: number;
  width: number;
  height: number;
  depth: Float32Array;
}

/**
 * Ocean field. Static-per-config arrays (tau/groupTau/speed/amp/depthEff) are rebuilt
 * when bathymetry/swell/tide change; dynamic arrays update every step.
 */
export interface OceanField {
  /** Wave travel-time from the offshore boundary, seconds (eikonal solve). */
  tau: Float32Array;
  /** Travel-time field for the set envelope, accumulated at group speed. */
  groupTau: Float32Array;
  /** Local shallow-water wave speed sqrt(g*depthEff), m/s. */
  speed: Float32Array;
  /** Local wave amplitude H0 * shoaling coefficient, metres. */
  amp: Float32Array;
  /** depth + tide, metres. */
  depthEff: Float32Array;
  /** cos(phase) at current time; 1 = crest. */
  cosPhase: Float32Array;
  /** Local breaking ratio (amp * setEnvelope) / depthEff. */
  ratio: Float32Array;
  /** 1 where the wave face is actively breaking this step. */
  breaking: Uint8Array;
  /** Whitewater amount 0..1 (persists and drifts shoreward). */
  foam: Float32Array;
}

export const SurferState = {
  PADDLE: 0,
  WAIT: 1,
  TAKEOFF: 2,
  RIDE: 3,
  WIPEOUT: 4,
  KICKOUT: 5,
  RETURN: 6,
} as const;
export type SurferState = (typeof SurferState)[keyof typeof SurferState];

export const SURFER_STATE_NAMES: Record<SurferState, string> = {
  [SurferState.PADDLE]: 'PADDLE',
  [SurferState.WAIT]: 'WAIT',
  [SurferState.TAKEOFF]: 'TAKEOFF',
  [SurferState.RIDE]: 'RIDE',
  [SurferState.WIPEOUT]: 'WIPEOUT',
  [SurferState.KICKOUT]: 'KICKOUT',
  [SurferState.RETURN]: 'RETURN',
};

/** One surfer's control input for one step. moveX/moveY in [-1, 1]. */
export interface SurferInput {
  moveX: number;
  moveY: number;
  takeoff: boolean;
}

export const NO_INPUT: SurferInput = { moveX: 0, moveY: 0, takeoff: false };

/**
 * Flat, data-oriented world state (same style as Crowded Lineup).
 * Surfer 0 is always the human player; 1..N are AI agents.
 */
export interface WorldState {
  config: WorldConfig;
  seed: number;
  rngState: number;
  time: number;
  stepCount: number;
  bathy: Bathymetry;
  swell: SwellParams;
  ocean: OceanField;
  numSurfers: number;
  surferX: Float32Array; // metres
  surferY: Float32Array;
  surferVX: Float32Array;
  surferVY: Float32Array;
  surferHeading: Float32Array; // radians, atan2(vy, vx) convention
  surferState: Uint8Array; // SurferState codes
  surferStateTime: Float32Array;
  surferStamina: Float32Array; // 0..100
  surferScore: Float32Array;
  surferCatches: Uint16Array;
  surferWipeouts: Uint16Array;
  surferRideTime: Float32Array; // current ride duration
  surferBestRide: Float32Array; // longest ride, seconds
  /** Time spent sitting in breaking/foam water while not riding. */
  surferInsideTime: Float32Array;
  /** Ride direction captured at takeoff. */
  rideDirX: Float32Array;
  rideDirY: Float32Array;
  /** world.time when the surfer last entered TAKEOFF (-Infinity if never). */
  surferTakeoffTime: Float32Array;
  /** Position at the moment the surfer last entered TAKEOFF (metres). */
  surferTakeoffX: Float32Array;
  surferTakeoffY: Float32Array;
  /**
   * Distance (m) to the nearest breaking cell, measured at the moment the
   * surfer entered TAKEOFF. Priority is judged on takeoff-time distances:
   * the breaking mask moves several metres within a second, so re-measuring
   * at adjudication time would judge a different wave than the one taken.
   */
  surferPeakDist: Float32Array;
  /** Count of wipeouts caused by dropping in on a surfer with priority. */
  surferDropins: Uint16Array;
}
