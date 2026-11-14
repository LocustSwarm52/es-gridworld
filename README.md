# es-gridworld

A small neural network learns to cross a maze in the browser, trained with Evolution Strategies instead of backpropagation. Every connection is clickable, so you can change a weight by hand, break the learned behaviour, and watch training repair it.

**Live demo:** `https://LocustSwarm52.github.io/es-gridworld/`

## What you can do

- **Train** the network and watch its fitness curve climb
- **Watch an episode**: the agent walks the maze while each node shades by its activation
- **Click any connection** to edit its weight with a slider; line color shows the sign and thickness shows the size
- **Shock the weights** to add large random noise, then train again and watch it recover
- **Randomize** to start over from scratch

## How it works

**The task.** A 5×5 grid with two wall segments. The agent starts in the top-left corner and must reach the goal in the bottom-right within 16 moves.

**The network** has 4 inputs, 6 hidden units, and 4 outputs (58 parameters):

- Inputs: the agent's x and y position, and its x and y distance to the goal, all scaled to between -1 and 1
- Hidden layer: 6 tanh units
- Outputs: a softmax over up, right, down, and left. The agent takes the most likely action.

**The reward.** Each move earns credit for getting closer to the goal and costs a little for taking time. Bumping a wall costs more. Reaching the goal is worth +10, and ending far away is penalized.

**Evolution Strategies** (`esStep()` in `es.js`), each training step:

1. Draw 20 random noise vectors the size of the weight vector.
2. Score the weights plus each noise vector, and minus each one (mirrored sampling, 40 trials in total).
3. Rank the scores and give each trial a utility between -0.5 and +0.5, so one lucky trial can't dominate. Tied scores share a rank.
4. Move the weights toward the noise directions that scored well: `θ ← θ + lr / (n·σ) · Σ utility_k · ε_k`.

No gradients are computed, which keeps the code short and lets you edit any weight without breaking a training pipeline.

**One design choice that mattered.** Scoring each trial only from the top-left start often left training stuck in a loop at a corner, solving about a quarter to half of random seeds. Scoring the average over all 18 open starting cells gives a smoother signal, and the network now solves the maze from about 95% of random starts (38 of 40 seeds in testing).

## Run it

Open `index.html` in a browser. To run the tests (Node 18 or newer):

```
npm test
```

The tests check the network's shape and outputs, that the maze is solvable, that training succeeds from several random seeds, and that a shocked network usually relearns.

## Files

| File | What it holds |
|---|---|
| `es.js` | Grid world, network, reward, and Evolution Strategies |
| `index.html` | The interface: grid, clickable network, and fitness chart |
| `test.js` | Tests |
