# AI Surf World — Anytime World Engine MVP

A top-down, AI-native surfing simulation game. The ocean is a dynamic world:
waves propagate over a bathymetry height field, shoal, refract and break where
the bottom tells them to. Player and rule-based AI surfers share the same
lineup, read the same ocean, and compete for waves.

**Play it:** https://suitzero.github.io/surf-world/

This is Milestone 1 of a long-term research project: build the surf game
first, then extract the common structure (an Anytime World Runtime) from
several working domains — not the other way around.

## How the ocean works (M1 approximation — not a high-fidelity physics claim)

- Bathymetry: 2D depth grid (128×96, 4 m cells), seeded generators —
  sandbar, straight beach, point break, reef.
- A travel-time field τ solves |∇τ| = 1/c with c = √(g·depth), seeded at the
  offshore boundary by the incoming swell (height, period, direction).
- Phase = ω(t−τ); shoaling amplification follows Green's law (d_ref/d)^¼.
- A cell breaks when it is near a crest and local height/depth ≥ 0.78 while
  the water just upwave is still subcritical — so the break line is a thin
  contour, whitewater (foam) sits shoreward of it, and because c varies
  along the crest the break point sweeps alongshore: peeling.
- Swell arrives in sets (slow amplitude envelope).

## Controls

- Paddle: WASD / arrow keys, or drag on the ocean (touch or mouse).
- Takeoff: paddle hard toward shore on a steep face, or press SPACE.
- World controls: bathymetry, swell height / period / direction, tide,
  AI surfer count (0–8), restart with a new seed.

## Tasks

- [x] S01 Game client (HUD, controls, measured FPS / sim ms)
- [x] S02 Bathymetry generator (4 seeded terrains)
- [x] S03 Basic ocean solver (propagation, shoaling, breaking, peeling)
- [x] S04 Wave visualization (depth map, crests, foam)
- [x] S05 Surfer dynamics (paddle, takeoff, ride, wipeout)
- [x] S06 Rule-based agents (catch waves using only current observable state)
- [ ] S07 High-fidelity teacher model hookup (SWASH / XBeach class)
- [ ] S08 Dataset pipeline
- [ ] S09 Neural Ocean Model
- [ ] S10 Neural inference runtime
- [ ] S11 Physics / Neural comparison
- [ ] S12 Hybrid correction
- [ ] S13 RL surfer training
- [ ] S14 Future rollout executor
- [ ] S15 Anytime compute control
- [ ] S16 Distributed workers
- [ ] S17 Public demo (this page)
- [ ] S18 Runtime refactoring (extract common interfaces from real code)

## Verification (tests/)

1. Same swell breaks at different locations on sandbar vs straight beach.
2. Reversing the swell angle sign flips the peel direction.
3. Wave speed in shallow water is slower than in deep water.
4. Bathymetry is deterministic per seed.
5. A rule-based AI catches at least one wave in a bounded headless run.
6. Whole-world determinism: same seed + inputs → identical trajectories.

## Develop

```
npm install
npm test        # vitest verification suite
npm run build   # tsc + vite build into dist/
npm run dev     # local dev server
```

Dependencies: vite, typescript, vitest. Nothing else.
