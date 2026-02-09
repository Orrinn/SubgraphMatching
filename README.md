# A Comprehensive Survey of Subgraph Matching

Source code for **A Comprehensive Survey of Subgraph Matching: \[Experiments & Analysis\]** (accepted to **SIGMOD 2026**).

## Abstract

Subgraph matching is a fundamental problem in graph analysis with a wide range of real-world applications. As subgraph matching techniques evolve, the existing mainstream filter-order-enumeration framework falls short in two aspects: (i) this filter-order-enumeration perspective overlooks an emerging line of compiler-based approaches with caching and validation-based orderings. (ii) The recent rise of complex pruning techniques has shifted the focus of core optimizations beyond filtering and enumeration. This paper advocates the need for a comprehensive survey that not only thoroughly discusses the compiler-based approaches (i.e., cache-based methods and their ordering techniques), but also reframes algorithm-level optimizations such that the role of pruning is adequately addressed.

This survey revisits 17 representative exploration-based subgraph matching methods—including both algorithm-level techniques and compiler-based ones—and establishes two optimization pillars, i.e., redundancy reduction and order generation, that can inherently summarize all these efforts. This newly established perspective permits us to systematically organize various optimization techniques and analyze how they interact with each other in the same implementation framework. Our contributions are: (i) Cache-, filter-, and prune-based strategies can remove both overlapping and different redundancies, sending our performance up to 1.81× faster than existing state-of-the-art (SOTA) settings, and (ii) heuristic and validation-based orderings, though grounded in fundamentally different design principles, often converge to similar behavior, leading to comparable performance in practice. Finally, (iii) we provide empirical guidance on when and how different strategies are most effective across diverse graph scenarios.


## How to use this repo

### Setup
This project depends on `fmt` and `Eigen3`. `Eigen3` is required by the `fPilos` implementation.

- For `fmt`: See the official [installation guide](https://fmt.dev/12.0/get-started/#installation) or the [GitHub repo](https://github.com/fmtlib/fmt). Installing from source is recommended.

- For `Eigen3`:
```bash
sudo apt update
sudo apt install -y libeigen3-dev
```

Optional: `spdlog` ([Github repo](https://github.com/gabime/spdlog)) is used to trace the matching process for debugging. You can skip it if you don’t need verbose logs.

### Complie

```bash
cmake -B build .
cd build
make -j
```

Compilation generates four binaries:
  - `overall_performance`: Overall performance comparison.
  - `optimizations_ablation`: Filter-based and cache-based ablations.
  - `prune_optimizations_ablation`: Prune-based ablations.
  - `order_ablations`: Ordering ablations.

### Execute

We provide one exmaple query (*n16_query*) and one exmaple data graph (*hprd*) under `data/query/` and `data/data/`. 

```bash
# Usage:
./bin/{binary} {data_graph} {query_name} [output_dir]

# Example: overall performance on hprd with n16_query
./bin/overall_performance hprd n16_query

# Example: intersection-reduction ablations on hprd with n16_query
# Here we use RI as the ordering strategy.
./bin/optimizations_ablation hprd n16_query
./bin/prune_optimizations_ablation hprd n16_query

# Example: ordering ablations on hprd with n16_query
# Here we use `fGQL` (GQL filtering) as the fixed optimization setting.
./bin/order_ablations hprd n16_query
```

## Input format

**Both query and data are unlabeled graphs**. 

Each graph starts with 't N M' where N is the number of vertices and M is the number of edges. A vertex and an edge are formatted as 'v VertexID LabelId Degree' and 'e VertexId VertexId' respectively. Note that we require that the vertex id is started from 0 and the range is [0,N - 1] where V is the vertex set. (See also: [RapidsAtHKUST/SubgraphMatching](https://github.com/RapidsAtHKUST/SubgraphMatching)
and [JackChuengQAQ/SubgraphMatchingSurvey](https://github.com/JackChuengQAQ/SubgraphMatchingSurvey))

Example:

```bash
t 5 6
v 0 0 2
v 1 1 3
v 2 2 3
v 3 1 2
v 4 2 2
e 0 1
e 0 2
e 1 2
e 1 3
e 2 4
e 3 4
```

**Of note**, `fPilos` also requires the precomputed eigenvalue index under `data/eigenIndex/`. For details on how the index is generated, please refer to the [original implementation](https://github.com/constantinosskitsas/Pilos-Subgraph-Matching).

## Code references

This repository builds on top of prior open-source subgraph matching implementations, including:

- [RapidsAtHKUST/SubgraphMatching](https://github.com/RapidsAtHKUST/SubgraphMatching)
- [JackChuengQAQ/SubgraphMatchingSurvey](https://github.com/JackChuengQAQ/SubgraphMatchingSurvey)

We also thank other open-source implementations referenced in the survey.