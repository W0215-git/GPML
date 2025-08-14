# GPML for Multimodal Medical Image Classification

This repository contains code for running Genetic Programming-based Multimodal Learning (GPML) experiments.

## Dataset Structure

The dataset should be placed inside a directory (e.g., `GAMMA/`) with the following structure:

```text
GAMMA/
├── fundus_train.npy
├── fundus_test.npy
├── oct_train.npy
├── oct_test.npy
├── labels_train.npy
└── labels_test.npy
```

### File Descriptions

- `fundus_train.npy`, `fundus_test.npy` — Fundus image modality (training and test sets)
- `oct_train.npy`, `oct_test.npy` — OCT image modality (training and test sets)
- `labels_train.npy`, `labels_test.npy` — Classification labels (primary)

## Running the Code

To run the main training script:

```bash
python Main.py --r $randomseed
```

Replace `$randomseed` with an integer seed (e.g., 1, 42).

## Configuration

Experiment settings are defined in the config JSON file.

### Example Configuration

```json
{
  "project": "Test",
  "dataDir": "GAMMA",
  "Class": 2,
  "r": 3,
  "population_size": 100,
  "generation": 50,
  "cxProb": 0.8,
  "mutProb": 0.19,
  "elitismProb": 0.01,
  "initialMinDepth": 2,
  "initialMaxDepth": 6,
  "maxDepth": 8,
  "tournament_size": 5,
  "Strategy": "voting",
  "K": 5,
  "w1": 0.6,
  "w2": 0.2,
  "w3": 0.2,
  "des": ""
}
```

### Configuration Parameters

You can change parameters such as:

- `dataDir` — Path to the dataset folder
- `Class` — Number of classes (currently supports binary = 2)
- `population_size` — GP population size
- `generation` — Number of generations to evolve
- `cxProb`, `mutProb` — Crossover and mutation probabilities
- `elitismProb` — Elitism probability
- `maxDepth` — Maximum tree depth

## Acknowledgment

This repository is built on reusing codes of https://github.com/YingBi92/BookCode