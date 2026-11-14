// Run with: node --test
const test = require('node:test');
const assert = require('node:assert');
const ES = require('./es.js');

test('network has 58 parameters and outputs a probability distribution', () => {
  assert.strictEqual(ES.N_PARAMS, 58);
  const theta = ES.randomParams(ES.rng(1));
  const { probs, hidden } = ES.forward(theta, ES.observe(0, 0));
  assert.strictEqual(hidden.length, 6);
  assert.strictEqual(probs.length, 4);
  assert.ok(Math.abs(probs.reduce((a, b) => a + b, 0) - 1) < 1e-9);
});

test('the goal is reachable around the walls', () => {
  // Breadth-first search over the grid.
  const q = [ES.START];
  const seen = new Set([ES.START.join()]);
  while (q.length) {
    const [x, y] = q.shift();
    for (const [dx, dy] of [[0, 1], [1, 0], [0, -1], [-1, 0]]) {
      const nx = x + dx, ny = y + dy;
      const k = nx + ',' + ny;
      if (nx < 0 || ny < 0 || nx >= ES.SIZE || ny >= ES.SIZE || ES.isWall(nx, ny) || seen.has(k)) continue;
      seen.add(k); q.push([nx, ny]);
    }
  }
  assert.ok(seen.has(ES.GOAL.join()));
});

test('Evolution Strategies learns to reach the goal from several seeds', () => {
  let solved = 0;
  for (const seed of [1, 2, 3, 4, 5]) {
    const rand = ES.rng(seed);
    let theta = ES.randomParams(rand);
    for (let i = 0; i < 300 && !ES.episode(theta).solved; i++) theta = ES.esStep(theta, rand).theta;
    if (ES.episode(theta).solved) solved++;
  }
  assert.ok(solved >= 4, `solved ${solved} of 5 seeds`);
});

test('a shock breaks a trained network and retraining recovers it', () => {
  let recovered = 0, tried = 0;
  for (const seed of [1, 2, 3, 4, 5, 6]) {
    const rand = ES.rng(seed);
    let theta = ES.randomParams(rand);
    for (let i = 0; i < 400 && !ES.episode(theta).solved; i++) theta = ES.esStep(theta, rand).theta;
    if (!ES.episode(theta).solved) continue;
    theta = ES.shock(theta, rand);
    if (ES.episode(theta).solved) continue; // the shock happened to miss
    tried++;
    for (let i = 0; i < 600; i++) {
      theta = ES.esStep(theta, rand).theta;
      if (ES.episode(theta).solved) { recovered++; break; }
    }
  }
  assert.ok(tried >= 3, 'shocks should usually break the policy');
  assert.ok(recovered >= Math.ceil(tried / 2), `recovered ${recovered} of ${tried}`);
});
