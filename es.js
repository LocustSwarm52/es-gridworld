// es-gridworld core: a grid world, a small neural network policy,
// and Evolution Strategies training. Browser (window.ES) and Node.

(function (root) {
  'use strict';

  // ---------------------------------------------------------------
  // Seeded random numbers, so runs can be reproduced.
  // ---------------------------------------------------------------
  function rng(seed) {
    let s = seed >>> 0;
    const next = () => {
      s = (s + 0x6d2b79f5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    next.gauss = () => {
      const u = Math.max(next(), 1e-12);
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * next());
    };
    return next;
  }

  // ---------------------------------------------------------------
  // Grid world. 5x5, start top-left, goal bottom-right, some walls.
  // Actions: 0 up, 1 right, 2 down, 3 left.
  // ---------------------------------------------------------------
  const SIZE = 5;
  const START = [0, 0];
  const GOAL = [4, 4];
  const WALLS = new Set(['1,1', '2,1', '3,1', '1,3', '2,3', '3,3']);
  const MOVES = [[0, -1], [1, 0], [0, 1], [-1, 0]];
  const ACTIONS = ['up', 'right', 'down', 'left'];
  const MAX_STEPS = 16;

  const isWall = (x, y) => WALLS.has(x + ',' + y);
  const inGrid = (x, y) => x >= 0 && y >= 0 && x < SIZE && y < SIZE;

  // What the network sees: position and offset to the goal, scaled to [-1, 1].
  function observe(x, y) {
    const s = (v) => (v / (SIZE - 1)) * 2 - 1;
    return [s(x), s(y), (GOAL[0] - x) / (SIZE - 1), (GOAL[1] - y) / (SIZE - 1)];
  }

  // ---------------------------------------------------------------
  // Network: 4 inputs -> 6 tanh hidden units -> 4 softmax outputs.
  // All weights live in one flat array so ES can perturb them.
  // ---------------------------------------------------------------
  const N_IN = 4, N_HID = 6, N_OUT = 4;
  const N_PARAMS = N_IN * N_HID + N_HID + N_HID * N_OUT + N_OUT; // 58

  // Index helpers into the flat parameter array.
  const idx = {
    w1: (i, h) => i * N_HID + h,
    b1: (h) => N_IN * N_HID + h,
    w2: (h, o) => N_IN * N_HID + N_HID + h * N_OUT + o,
    b2: (o) => N_IN * N_HID + N_HID + N_HID * N_OUT + o,
  };

  function forward(theta, input) {
    const hidden = new Array(N_HID);
    for (let h = 0; h < N_HID; h++) {
      let z = theta[idx.b1(h)];
      for (let i = 0; i < N_IN; i++) z += input[i] * theta[idx.w1(i, h)];
      hidden[h] = Math.tanh(z);
    }
    const logits = new Array(N_OUT);
    for (let o = 0; o < N_OUT; o++) {
      let z = theta[idx.b2(o)];
      for (let h = 0; h < N_HID; h++) z += hidden[h] * theta[idx.w2(h, o)];
      logits[o] = z;
    }
    const m = Math.max(...logits);
    const exps = logits.map((z) => Math.exp(z - m));
    const sum = exps.reduce((a, b) => a + b, 0);
    return { hidden, probs: exps.map((e) => e / sum) };
  }

  const argmax = (a) => a.reduce((best, v, i) => (v > a[best] ? i : best), 0);

  // Play one episode with the greedy (most likely) action each step.
  // Returns the reward and the path, for drawing.
  function episode(theta, start = START) {
    let [x, y] = start;
    const path = [[x, y]];
    const dist = (px, py) => Math.abs(GOAL[0] - px) + Math.abs(GOAL[1] - py);
    let reward = 0;
    for (let t = 0; t < MAX_STEPS; t++) {
      const a = argmax(forward(theta, observe(x, y)).probs);
      const nx = x + MOVES[a][0];
      const ny = y + MOVES[a][1];
      if (!inGrid(nx, ny) || isWall(nx, ny)) {
        reward -= 0.3; // bumped into a wall or the edge
      } else {
        reward += 0.5 * (dist(x, y) - dist(nx, ny)); // progress toward the goal
        x = nx; y = ny;
      }
      path.push([x, y]);
      reward -= 0.05; // small cost per step favours short paths
      if (x === GOAL[0] && y === GOAL[1]) return { reward: reward + 10, path, solved: true };
    }
    return { reward: reward - 0.5 * dist(x, y), path, solved: false };
  }

  // Training score: the average over every open starting cell. Scoring
  // from many starts gives a much smoother signal than one start alone.
  const STARTS = [];
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      if (!isWall(x, y) && !(x === GOAL[0] && y === GOAL[1])) STARTS.push([x, y]);
    }
  }
  function fitness(theta) {
    let total = 0;
    for (const s of STARTS) total += episode(theta, s).reward;
    return total / STARTS.length;
  }

  // The "shock" button: add large random noise to every weight.
  function shock(theta, rand, scale = 1) {
    return theta.map((w) => w + rand.gauss() * scale);
  }

  function randomParams(rand, scale = 0.5) {
    return Array.from({ length: N_PARAMS }, () => rand.gauss() * scale);
  }

  // ---------------------------------------------------------------
  // Evolution Strategies (OpenAI-style, with mirrored sampling):
  // try small random changes to the weights, score each one, and move
  // the weights toward the changes that scored better. No gradients.
  // ---------------------------------------------------------------
  function esStep(theta, rand, opts = {}) {
    const { pairs = 20, sigma = 0.5, lr = 0.5 } = opts;
    const noises = [];
    const scores = [];
    for (let k = 0; k < pairs; k++) {
      const eps = Array.from({ length: N_PARAMS }, () => rand.gauss());
      for (const sign of [1, -1]) {
        const trial = theta.map((w, i) => w + sign * sigma * eps[i]);
        noises.push(sign === 1 ? eps : eps.map((e) => -e));
        scores.push(fitness(trial));
      }
    }
    // Rank-normalise the scores so one lucky trial cannot dominate.
    // Tied scores share their average rank, so ties add no noise.
    const order = scores.map((s, i) => [s, i]).sort((a, b) => a[0] - b[0]);
    const util = new Array(scores.length);
    for (let a = 0; a < order.length;) {
      let b = a;
      while (b + 1 < order.length && order[b + 1][0] === order[a][0]) b++;
      const rank = (a + b) / 2;
      for (let k = a; k <= b; k++) util[order[k][1]] = rank / (scores.length - 1) - 0.5;
      a = b + 1;
    }

    const next = theta.slice();
    for (let p = 0; p < N_PARAMS; p++) {
      let g = 0;
      for (let k = 0; k < noises.length; k++) g += util[k] * noises[k][p];
      next[p] += (lr / (noises.length * sigma)) * g;
    }
    return { theta: next, meanScore: scores.reduce((a, b) => a + b, 0) / scores.length };
  }

  const api = {
    rng, SIZE, START, GOAL, WALLS, ACTIONS, MAX_STEPS, isWall,
    N_IN, N_HID, N_OUT, N_PARAMS, idx, observe, forward, episode, fitness, STARTS, shock, randomParams, esStep,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ES = api;
})(typeof window !== 'undefined' ? window : globalThis);
