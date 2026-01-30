# Learning to Learn: Memory-Guided Evolution of Computation Graphs via Self-Compressing Fitness-Regularized Latent Space

## Contents
- [Existing Literature](#existing-literature)
  - [Neural Architecture Search](#neural-architecture-search)
- [Proposed Solution](#proposed-solution)
  - [Multi-Objective Fitness (Pareto Optimization)](#multi-objective-fitness--pareto-optimization-)
  - [Canonicalized Metrics for Surrogate Guidance](#canonicalized-metrics-for-surrogate-guidance)
  - [Stabilizing Graph Attribute Decoding](#stabilizing-graph-attribute-decoding)
  - [Handling Invalid Guided Offspring](#handling-invalid-guided-offspring)
  - [Generative Cross-Species Crossover (Graph-VAE)](#generative-cross-species-crossover--graph-vae-)
- [Other Papers that Might be Useful](#other-papers-that-might-be-useful)
- [Future Ideas](#future-ideas)
- [Maybe Ideas](#maybe-ideas)
- [References](#references)

## Existing Literature
### Neural Architecture Search
- Limitations of current methods
  - Limited adaptability: The final solution is static (even ones with continued online learning have a static learning paradigm)
  - Most focus only on topology and/or weights
     - “Such experimental trials convinced us that to solve the [Neural Architecture Search] dilemma, the connectionist paradigm alone is not adequate.” [1](#references)
  - Weight sharing speeds up the search by reducing the size of the search space but "there is evidence that [weight sharing] inhibits the search for optimal architectures (Yu et al., 2020)" though this would only apply under some circumstances. [5](#references)
  - Can take a while to find solution
     - [5](#references) addresses this by linearizing the loss function of untrained networks at the data points in each batch and using Kendall’s Tau correlation coefficient, which measures the strength and direction of association that exists between two variables, to determine that an initialized network is not worth training.
     It is uncertain whether this generalizes to harder tasks.
     It also remains to be seen whether it can be used for an unsupervised context.
     It might be adaptable to reinforcement learning but that has not been shown and could prove unstable due to the general sparsity of signal in the reinforcement context.
     It's also unclear if this method can be adapted to spiking neural networks.

## Proposed Solution
### Initial Population
`computation_graphs/optimizers/` includes TorchScript optimizers that seed generation 0. The following algorithms are available out of the box:

- Plain SGD family: `gradient_descent_backprop.py`, `momentum_sgd_backprop.py`, `nesterov_sgd_backprop.py`, `sgdw_backprop.py`
- Classic adaptive methods: `adagrad_backprop.py`, `rmsprop_backprop.py`, `adagrad_norm_backprop.py`, `rmsprop_norm_backprop.py`
- Adam-family variants: `adam_backprop.py`, `adamw_backprop.py`, `radam_backprop.py`, `qhadam_backprop.py`, `yogi_backprop.py`, `ada_belief_backprop.py`, `lamb_backprop.py`
- Other adaptive/normalized optimizers: `adafactor_backprop.py`, `adan_backprop.py`, `adanorm_backprop.py`
- Curvature- or Hessian-aware methods: `adahessian_backprop.py`, `lion_backprop.py`, `scaled_signsgd_backprop.py`
- Lookahead / proximal hybrids: `lookahead_adam_backprop.py`, `lookahead_rmsprop_backprop.py`, `ftrl_proximal_backprop.py`
- Resilient backprop family: `rprop_backprop.py`, `irprop_plus_backprop.py`

To generate the corresponding .pt files, run this command in the main directory: `find computation_graphs/optimizers -name '*.py' -print0 | xargs -0 -n1 python3.10`

### Experiment Tracking and CLI options
Run `main.py` directly to evolve optimizers. The script now exposes switches for MLflow tracking and general configuration, so you can keep experiments reproducible:

```
python3 main.py \
  --config-file neat-config \
  --num-generations 250 \
  --enable-mlflow \
  --mlflow-experiment learning-to-learn \
  --mlflow-run-name warmup_run \
  --mlflow-tag stage=warmup --mlflow-tag dataset=cifar10
```

Key flags:

- `--config-file`: choose an alternate NEAT config.
- `--num-generations`: change the evolutionary horizon (default 1000).
- `--enable-mlflow`: turn on MLflow streaming. Pair it with `--mlflow-tracking-uri`, `--mlflow-experiment`, `--mlflow-run-name`, `--mlflow-tag KEY=VALUE`, and `--mlflow-nested` to match your tracking server layout.

When MLflow is enabled, the run logs population best/mean/worst fitnesses, species counts, genome complexity per generation, final winner metrics, the NEAT config artifact, and a JSON summary of invalid guided-offspring reasons. It now also captures:

- OnlineTrainer epoch summaries (adjacency/attribute reconstruction, KL terms, fitness loss, and totals) so the `Epoch … Loss terms per batch` lines end up in the MLflow run as metrics + `logs/progress.log` text.
- Per-metric fitness predictor losses (`trainer_metric_<metric_name>` metrics plus the CSV/HTML artifacts) so you can see which task objectives dominate each epoch.
- Guided offspring production stats (`guided_children_*` metrics for requested/created totals and invalid counts per reason) logged once per generation alongside your trainer curves.
- Genetic-distance statistics, compatibility-threshold changes, and a per-generation species table (same columns as the NEAT stdout reporter) under `species/generation_*.json`.

This mirrors what shows up in stdout while giving you a permanent experiment record.

### Multi-Objective Fitness (Pareto Optimization)

A central innovation is the use of **Pareto-based multi-objective optimization** for evaluating and selecting candidate networks.
Each network (or "individual") in the population is evaluated on multiple objectives rather than a single metric.
One set of objectives reflect task-specific performance and another reflects computational cost.
Instead of combining these objectives into a single weighted fitness score, the algorithm employs **Pareto ranking** via non-dominated sorting: an individual is considered fitter if it is *Pareto-superior*-that is, better in at least one objective without being worse in any other-compared to others in the population ([53](#references)).

Using Pareto-based selection yields a diverse **Pareto front** of solutions, each representing a different balance of objectives (for example, one network might be extremely simple but moderately accurate, while another is more complex but achieves higher accuracy).
This approach has several advantages for evolving learning systems:

* **Robust, Generalizable Solutions:**
Multi-objective evolution tends to produce more robust models than optimizing a single metric.
The algorithm doesn't commit to one arbitrary trade-off; instead, it preserves a spectrum of high-performing solutions.
Researchers can then analyze this Pareto front for insights into how different architectures trade off performance vs. complexity ([53](#references)).
* **Favoring Minimal Architectures:**
By explicitly treating computational cost as an objective to minimize, the evolution **naturally favors minimal efficient networks** that align with the Minimum Description principle-the best solution is the simplest one that explains the data ([55](#references)).
The smallest networks that still perform well can be seen as **minimum description estimates (MDEs)** of the task.
These MDEs are not just efficient; they are also scientifically interpretable, as they highlight the core components necessary to implement a given function or behavior.
* **No Manual Tuning of Trade-offs:**
Pareto ranking removes the need to hand-tune weights between objectives (such as how much to penalize complexity).
Instead of a single "optimal" network according to a weighted sum, a set of Pareto-optimal networks is obtained that captures different compromises.
This is especially useful in research contexts, where one might prefer simpler models for interpretation unless complexity is absolutely required for performance ([53](#references)).

In summary, multi-objective evolutionary selection ensures that the project optimizes not only for how well a network learns, but also for how elegant or tractable its design is.
This keeps the search grounded, preventing bloated solutions and steering evolution toward general-purpose networks that are both **high-performing and low-complexity**.

### Canonicalized Metrics for Surrogate Guidance

The Pareto sorter stays weight-free, but the surrogate (Graph-VAE + predictor) must reason about heterogeneous metrics whose raw scales differ by orders of magnitude.
Two mechanisms keep the guided offspring search well-behaved:

- **Signed log-space distances:**
Each metric advertises a best value (typically 0).
Before computing the surrogate loss raw scores are converted to `sign(best_value) * log1p(|best_value|)`, compressing large ranges so accuracy, time, and memory all yield comparable gradients.
- **Per-metric guidance weights:**
Metrics also expose a guidance weight that scales their contribution in the surrogate/predictor losses and the latent optimization loop.
Pareto ranking remains unbiased; weights only affect how the surrogate prioritizes improvements when generating children.

### Handling Invalid Guided Offspring

The guided decoder sometimes samples latent points that produce unusable computation graphs (e.g., no edges or incompatible tensor shapes), particularly in the first few generations.
To keep these failures from overwhelming the surrogate model:

* Store invalid graph/metric pairs separately from the valid ones with a fixed penalty.
* Mix only a small, generation-proportional fraction of the invalid graphs into each training epoch (capped at 20% of the valid graphs), so early generations focus on valid data while later ones still learn which latent regions to avoid

This subsampling keeps the decoder from collapsing onto invalid DAGs while still providing a clear gradient signal to steer it back toward feasible graphs.

### Stabilizing Graph Attribute Decoding

To avoid decoder stalls from attr-name loops that never received a termination signal, teacher forcing is used for those sequences:

- The shared attribute vocabulary assigns explicit `<SOS>`/`<EOS>` tokens and exposes helpers that serialize each node’s dynamic attributes into deterministic name sequences.
  These targets supervise the attr-name GRU whenever ground-truth graphs exist.
- The graph decoder optionally consumes those targets, drives the GRU with embedded ground-truth tokens, and accumulates a per-step cross-entropy loss.
- The trainer groups batched node attributes, builds the token targets, passes them through the guide model, and blends the decoder’s cross-entropy into the feature reconstruction term.

### Generative Cross-Species Crossover (Graph-VAE)

Another key innovation is a Variational Autoencoder (VAE) that enables unrestricted structural recombination of neural network architectures.
Traditional neuroevolution methods like NEAT use crossover within the same "species" of networks and rely on aligning genes (nodes and connections) based on historical markers to recombine two parents.
Those methods struggle to recombine vastly different topologies because the correspondence between parts of two very different networks is ambiguous.
The approach addresses this by **learning a continuous encoding of network graphs**, allowing any two networks-even of entirely different designs-to **mate** in a meaningful way ([54](#references)).

The implemented module (called *SelfCompressingFitnessRegularizedDAGVAE*, in [`search_space_compression.py`](fsearch_space_compression.py)) acts as a **generative recombination mechanism** or a "cross-species mating system" for networks:

* **Graph Encoding and Decoding:**
The VAE is trained to take arbitrary computation graphs and encode them into a continuous latent vector space. It can then decode a latent vector back into a network graph.
This provides a common representation for all networks, regardless of their topology ([54](#references)).
* **Fitness-Regularized Compression:**
The VAE's training is not just a standard graph autoencoding.
It is **regularized with fitness prediction**: part of the VAE's objective is to predict the network's performance and complexity from the latent encoding.
This means the VAE is encouraged to organize the latent space such that important features (those that correlate with high fitness) are captured in the representation.
Dimensions of the latent vector that do not contribute to explaining variation in performance tend to be pruned out automatically (using techniques like Automatic Relevance Determination).
The result is a compressed, fitness-informed search space: a smooth landscape where distances reflect meaningful differences in network capability.
* **Cross-Species Mating via Latent Interpolation:**
Once networks are encoded in this latent space, **any two networks** can be recombined by interpolating or randomly mixing their latent vectors and then decoding the result.
In other words, the VAE enables **cross-species crossover**-two parent networks from entirely different niches or architectures can produce an offspring network.
This generative crossover bypasses the historical gene-matching problem that NEAT faced without needing to explicitly align nodes or connections between parents.
* **Sharing Innovations Across Lineages:** This mechanism effectively breaks down barriers between separate evolutionary lineages.
Useful innovations that arise in one species can be transferred to very different architectures. Because a portion of each generation's offspring are generated via latent-space recombination, **new architectures can emerge that combine traits from wildly different ancestors**. This helps inject novel structural patterns that pure mutation or traditional crossover (limited to similar parents) might never produce.
* **Expanded Exploration with Guided Search:** The VAE-driven crossover significantly **expands the exploration** of the search space while still being guided towards promising regions (due to the fitness-informed latent space).
It serves as a kind of **surrogate model** that directs evolution: by sampling in latent space, it effectively samples networks that are expected to be high-performing or at least valid.
This compresses the combinatorially vast search space of all possible networks into a more tractable form.
The net effect is **increased diversity** of candidate solutions and the ability to discover innovative network motifs that conventional genetic operators might miss ([54](#references)).

Importantly, the evolutionary algorithm **combines** this Graph-VAE crossover with more traditional NEAT-style mating within species.
In practice, this means there are two crossover pathways: (1) standard crossover between similar individuals (preserving fine-tuned structures within a species), and (2) occasional **graph-VAE generated offspring** that mix across species.
This balance ensures both **exploitation and exploration**: the population can refine known good solutions while still injecting radically new variations.

### Penalization of Invalid Offspring

To prevent invalid or non-operative graphs from biasing the surrogate or inflating Pareto scores, every genome passes explicit validity filters before it is evaluated.
Any sample that fails (decodes to an empty DAG, cannot be rebuilt, or produces an optimizer that leaves model parameters unchanged) is assigned a deterministic penalty vector: each fitness metric is set to ±10^6 depending on its objective direction.
These penalties propagate into the population’s fitness log.
Invalid graphs are omitted from normalizations for pareto fronts, etc.

### Other explanations to incoporate (!!TODO)
*Guided latent retries.* To keep promising latents alive, the decoder retains a non-empty graph seen during each child’s decode attempts if available and jitters subsequent retries around that anchor rather than restarting from the original latent.
Empirically this turns “almost valid” intermediate graphs into stepping stones, increasing the likelihood that at least one decode per latent survives the structural filters (implementation: [`population.py:260-340`](population.py)).

Immediately after the seed generation is evaluated, the self-compressing autoencoder (SCAE) undergoes a dedicated 100-epoch warm-up.

Linear ramp up of percentage of population created via guided mechanism to provide SCAE with more training examples before relying so heavily on it (mitigation of mode collapse).

Adaptive metric scaling for the fitness predictor uses a learned log-variance per metric inside the [`FitnessPredictor`](search_space_compression.py).
Each metric’s squared canonical error is modulated by `exp(-s_i)` and regularized by `s_i`, where `s_i` is the head’s log variance parameter (initialized at 0).
This keeps gradients for wildly different metrics (e.g., memory cost vs. AU task reward) much closer to the same numeric range without relying on per-generation normalization and still allows `metric_guidance_weights` to express explicit preferences.

Tether the latent search to the posterior (L2 penalty on deviation from the encoded seed) so that z_g stays in regions where the decoder was trained to emit real graphs.

## Other Papers that Might be Useful
- [On the Relationship Between Variational Inference and Auto-Associative Memory](https://arxiv.org/pdf/2210.08013.pdf)
  - "In order to improve the memory capacity, modern Hopfield networks [22, 21, 8] propose several variants of the energy function using polynomial or exponential interactions.
  Extending these models to the continuous case, [30] proposed the Modern Continuous Hopfield Network (MCHN) with update rules implementing self attention, that they relate to the transformer model [36].
  In [26], the authors introduce a general Hopfield network framework where the update rules are built using three components: a similarity function, a separation function, and a projection function."
  - "It has been shown that overparameterized auto-encoders also implement AM [28, 33].
  These methods embed the stored patterns as attractors through training, and retrieval is performed by iterating over the auto-encoding loop."
- [Multifactorial Evolutionary Algorithm with Online Transfer Parameter Estimation: MFEA-II](https://www.researchgate.net/profile/Abhishek-Gupta-17/publication/331729696_Multifactorial_Evolutionary_Algorithm_With_Online_Transfer_Parameter_Estimation_MFEA-II/links/5c98495892851cf0ae95ec75/Multifactorial-Evolutionary-Algorithm-With-Online-Transfer-Parameter-Estimation-MFEA-II.pdf)
- [STCA: Spatio-Temporal Credit Assignment with Delayed Feedback in Deep Spiking Neural Networks](https://www.ijcai.org/proceedings/2019/0189.pdf)
- [Sparse Distributed Memory using N-of-M Codes](apt.cs.manchester.ac.uk/ftp/pub/apt/papers/NofMnnV3.pdf)
- [A discrete time neural network model with spiking neurons: II: Dynamics with noise](https://arxiv.org/abs/1709.06206)
- [The Information Bottleneck Problem and Its Applications in Machine Learning](https://arxiv.org/pdf/2004.14941.pdf)
  - “The information bottleneck (IB) theory recently emerged as a bold information-theoretic paradigm for analyzing DL systems.
  Adopting mutual information as the figure of merit, it suggests that the best representation T should be maximally informative about Y while minimizing the mutual information with X.” (a.k.a. compression)
- [Neurons detect cognitive boundaries to structure episodic memories in humans](https://authors.library.caltech.edu/107546/4/41593_2022_1020_MOESM1_ESM.pdf)
- [Why think step-by-step? Reasoning emerges from the locality of experience](https://arxiv.org/pdf/2304.03843.pdf)
- [Accurate online training of dynamical spiking neural networks through Forward Propagation Through Time](https://arxiv.org/abs/2112.11231#:~:text=Here%2C%20we%20show%20how%20a,regularized%20risk%20on%20the%20loss.)
- [Dynamics-Aware Unsupervised Discovery of Skills](https://arxiv.org/abs/1907.01657)
- [Frahmentation Instability in Aggregating Systems](https://www.sciencedirect.com/science/article/abs/pii/S0378437122000930)
- [Rethinking the performance comparison between SNNs and ANNs](https://www.sciencedirect.com/science/article/abs/pii/S0893608019302667)
- [A survey on computationally efficient neural architecture search](https://www.sciencedirect.com/science/article/pii/S2949855422000028)
- [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)
- [Interpreting neural computations by examining intrinsic and embedding dimensionality of neural activity
](https://arxiv.org/abs/2107.04084)
  - "Task-relevant computations usually depend on structured neural activity whose dimensionality is lower than that of the state space"
  - "we refer to the number of Euclidean dimensions that are required to capture a given fraction of the structured activity as the embedding dimensionality of the data"
  - "While the embedding dimensionality carries information about structure, because of an assumption of linearity, it does not in general correspond to the number of independent variables needed to describe the data ... we call the number of independent variables the intrinsic dimension, and we refer to each independent variable describing the neural activity as a latent variable."
  - talks about instances that have been shown to have higher embedding than intrinsic dimensionality
  - "we focus on the key question of how latent variables and their embedding might relate to task variables and the underlying computations."
  - "the intrinsic dimensionality of neural activity is largely determined by three sources of information: (1) incoming stimuli, (2) ongoing movements, and (3) the multitude of latent variables that characterize prior experiences and future expectations."
  - "In early sensory areas, the intrinsic dimensionality is expected to be strongly associated with incoming stimuli."
  - "signals in the primary motor cortex seem to carry information about the upcoming movement without regard to higher order latent variables ... the intrinsic dimensionality in the late stages of the motor system seem to be strongly tied to ongoing movements"
  - "Although the general principles that govern embedding dimensionality are not known, several computationally inspired hypotheses have been proposed.
  One dominant hypothesis is that the information in any given brain area is extracted by other areas and peripheral systems through a linear readout.
  A linear readout scheme is a weighted average of the activity across neurons and has an intuitive geometric interpretation: it is a projection of the neural activity in the full Euclidean state space along a specific direction.
  One common strategy to evaluate this hypothesis is to quantify the degree to which a linear decoder can extract desired information from population activity in a brain area."
  - "An extension of this hypothesis is that embedding across a population of neurons is organized such that different linear decoders can extract information about different task-relevant variables without interference.
  For example, movement and movement preparation signals in monkeys’ motor cortex reside in orthogonal subspaces [53].
  This organization could ensure that preparation does not trigger movement [54,55], and may help protect previous motor memories [56]. Similar principles might govern interactions across cortical areas, by confining to orthogonal subspaces information that is private to an area and information that is communicated [57,58] ... We note however, that despite strong enthusiasm, direct evidence that the brain processes and/or communicates information through orthogonal linear decoders is wanting."
  - "In general, high-dimensional embeddings with more degrees of freedom can facilitate extraction of task-relevant variables without interference (Rigotti et al. 2013; Cayco-Gajic et al. 2017; Cayco-Gajic and Silver 2019; Litwin-Kumar et al. 2017; Lanore et al. 2021).
  However, embedding information in arbitrarily high dimensional subspaces can have adverse effects for generalization [64].
  To improve generalization, embeddings have to be appropriately constrained to capture the structural relationships and inherent invariances in the environment [10,38,65].
  A theory for the ensuing trade-offs has been recently developed for the case where the activity is organized in multiple disjoint manifolds corresponding to different categories [66] and applied to interpret representations in the visual system [67] and in deep networks [68].
  It is also possible to organize information embedding such that certain linear projection reflect information in more abstract form and therefore enable generalization, while others enable finer discrimination [15]."
  - "The utility of linear decodability however, becomes less clear for intermediate stages of information processing in higher brain areas that carry information about latent variables that support flexible mental computations.
  While an experimenter may apply linear decoders to find information about a hypothesized latent variable in a certain brain area, there is no a priori reason to assume that the brain relies on such decoders."
  - "For instance, ring-like manifolds, on which activity is represented by a single angular latent variable, can emerge from only weak structure in the connectivity"
- [Model-agnostic Measure of Generalization Difficulty](https://arxiv.org/abs/2305.01034)
- [Adaptive Inference through Early-Exit Networks: Design, Challenges and Directions](https://arxiv.org/abs/2106.05022)
  - very interesting paper about progressive inference using an exit policy based on context
  - probably use the "vanilla backbone networks, enhanced with early exits along their depth" approach because modularity would be useful for the gene representations
    - "when disentangling the backbone network’s design from the early exits, one can have the flexibility of lazily selecting the architecture of the latter ones"
  - "existence of residual connections spanning across early exits can help generalisability of the network"
  - "maintaining multiple feature size representations, can prove detrimental in terms of model footprint"
  - even though using a non-uniform architecture for the early exits increases the search space, it also allows for tradeoffs between "The number (and type) of exit-specific layers accuracy vs. their overhead"
  - "too many early classifiers can negatively impact convergence when training end-to-end"
  - equidistant vs variable distance "decision depends on the use-case, the exit rate and the accuracy of each early exit"
  - "inter-exit distance is not actual 'depth', but can be quantified by means of FLOPs or parameters in the network"
  - train network and early exits together
    - "joint loss function is shaped which sums intermediate and the last output losses (𝐿(𝑖)) in a 𝑡𝑎𝑠𝑘 weighted manner (Eq. 1) and then backpropagates the signals to the respective parts of the network"
    - "accuracy of this approach can be higher both for the intermediate (𝑦𝑖<𝑁) and the last exit (𝑦𝑁)" but "not guaranteed due to cross-talk between exits"
    - "interplay of multiple backpropagation signals and the relative weighting (𝑤𝑖) of the loss components" finnicky w.r.t. "enabl[ing] the extraction of reusable features across exits"
  - train separately
    - first train backbone then freeze it, place and train the exit policies
    - "means that each exit is only fine-tuning its own layers and does not affect the convergence of the rest of the network"
    - no "cross talk between classifiers nor need to hand-tune the loss function", which allows "more exit variants [to] be placed at arbitrary positions in the network and be trained in parallel, offering scalability in training while leaving the selection of exit heads for deployment time"
    - "more restrictive in terms of degrees of freedom on the overall model changes, and thus can yield lower accuracy than an optimised jointly trained variant."
  - can use knowledge distillation setup where "the student 𝑖 is typically an early exit and the teacher 𝑗 can be a subsequent or the last exit"
    - hyperparameters:
      - distillation temperature (𝑇) "controls how 'peaky' the teacher softmax (soft labels) should be"
      - alpha (𝛼) "balances the learning objective between ground truth (𝑦) and soft labels (𝑦𝑗)"
  - to deal with non-IID and inference-context-specific variation, can customize early exits "while retaining the performance of the last exit in the source global domain"
    - can still be used in conjunction with knowledge distillation
  - 3 main deployment methods: up to specific exit, full adaptive inference (each sample exits early depending on difficulty), budgeted adaptive inference ("maximise the throughput of correct predictions" given i.e. a total latency budget)
  - rule-based exiting "need[s] to manually define an arbitrary threshold"
  - learned early exiting can be modeled with or without independence from previous states of exit decisions
- [Network of Evolvable Neural Units: Evolving to Learn at a Synaptic Level](https://arxiv.org/abs/1912.07589)
  - uses 4-gate recurrent nodes (reset, update, cell, output)
  - each node uses same shared gate parameters
  - each gate is a single layer feedforward network with nonlinear activation that has the same number of outputs as there are memory states
  - this reduces the required chromosomes to two
- [Leveraging dendritic properties to advance machine learning and neuro-inspired computing](https://arxiv.org/abs/2306.08007)
- [Dendrites and Efficiency: Optimizing Performance and Resource Utilization](https://arxiv.org/abs/2306.07101)
- [Faith and Fate: Limits of Transformers on Compositionality](https://arxiv.org/pdf/2305.18654.pdf)
- [Building transformers from neurons and astrocytes](https://www.pnas.org/doi/epdf/10.1073/pnas.2219150120)

---

## References
1. [An Empirical Review of Automated Machine Learning](https://www.mdpi.com/2073-431X/10/1/11#sec3-computers-10-00011)
2. [AutoML: A survey of the state-of-the-art](https://arxiv.org/pdf/1908.00709.pdf?arxiv.org)
3. [Automated machine learning: Review of the state-of-the-art and opportunities for healthcare](https://www.sciencedirect.com/science/article/pii/S0933365719310437)
4. [A Systematic Literature Review of the Successors of “NeuroEvolution of Augmenting Topologies”](https://direct.mit.edu/evco/article/29/1/1/97341/A-Systematic-Literature-Review-of-the-Successors)
5. [Neural Architecture Search without Training](https://arxiv.org/abs/2006.04647)
6. [Spiking Neural Networks and online learning: An overview and perspectives](https://arxiv.org/pdf/1908.08019.pdf)
7. [Supervised learning in spiking neural networks: A review of algorithms and evaluations](https://www.researchgate.net/profile/Xiangwen-Wang-3/publication/339481763_Supervised_learning_in_spiking_neural_networks_A_review_of_algorithms_and_evaluations/links/5e70ea344585150a0d167d97/Supervised-learning-in-spiking-neural-networks-A-review-of-algorithms-and-evaluations.pdf)
8. [A review of learning in biologically plausible spiking neural networks](https://orca.cardiff.ac.uk/id/eprint/126388/1/Aboozar%20Manuscript.pdf)
9. [Training Spiking Neural Networks Using Lessons From Deep Learning](https://arxiv.org/abs/2109.12894)
10. [Deep Residual Learning in Spiking Neural Networks](https://proceedings.neurips.cc/paper/2021/file/afe434653a898da20044041262b3ac74-Paper.pdf)
11. [Event-based backpropagation can compute exact gradients for spiking neural networks](https://arxiv.org/pdf/2009.08378.pdf)
12. [Symmetric spike timing-dependent plasticity at CA3–CA3 synapses optimizes storage and recall in autoassociative networks](https://www.nature.com/articles/ncomms11552)
13. [Attention Spiking Neural Networks](https://arxiv.org/abs/2209.13929)
14. [Foundations and Modeling of Dynamic Networks Using Dynamic Graph Neural Networks: A Survey](https://ieeexplore.ieee.org/abstract/document/9439502)
15. [The synchronized dynamics of time-varying networks](https://arxiv.org/pdf/2109.07618.pdf)
16. [Multimodality in Meta-Learning: A Comprehensive Survey](https://arxiv.org/pdf/2109.13576)
17. [Meta-Learning in Neural Networks: A Survey](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9428530)
18. [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Neural networks](https://arxiv.org/pdf/1703.03400.pdfhttps://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9428530)
19. [Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206)
20. [A Generalist Agent](https://openreview.net/pdf?id=1ikK0kHjvj)
21. [PaLM-E: An Embodied Multimodal Language Model](https://arxiv.org/abs/2303.03378)
22. [Attention is all you need](https://arxiv.org/abs/1706.03762)
23. [Investigating the Limitations of Transformers with Simple Arithmetic Tasks](https://arxiv.org/abs/2102.13019)
24. [Transformers In Vision: A Survey](https://arxiv.org/pdf/2101.01169.pdf)
25. [Coinductive guide to inductive transformer heads](https://www.arxiv-vanity.com/papers/2302.01834/)
26. [Addressing Some Limitations Of Transformers With Feedback Memory](https://arxiv.org/abs/2002.09402)
27. [Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?](https://arxiv.org/abs/2202.12837)
28. [Why Can GPT Learn In-Context? Language Models Secretly Perform Gradient Descent as Meta-Optimizers](https://arxiv.org/pdf/2212.10559.pdf)
29. [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
30. [Emergence of Scale-free Graphs in Dynamical Spiking Neural Networks](https://filip.piekniewski.info/stuff/papers/poster2007.pdf)
31. [Learning place cells, grid cells and invariances with excitatory and inhibitory plasticity](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5927772/)
32. [On the approximation by single hidden layer feedforward neural networks with fixed weights](https://arxiv.org/pdf/1708.06219.pdf)
33. [Compressive Transformers For Long-Range Sequence Modelling](https://arxiv.org/pdf/1911.05507.pdf)
34. [Compressed Learning: A Deep Neural Network Approach](https://arxiv.org/pdf/1610.09615.pdf)
35. [Rethinking Lossy Compression: The Rate-Distortion-Perception Tradeoff](proceedings.mlr.press/v97/blau19a/blau19a.pdf)
36. [Lecture from Stanford by Jack Rae - Compression for AGI](https://www.youtube.com/live/dO4TPJkeaaU?feature=share)
37. [How learning unfolds in the brain: toward an optimization view](https://www.sciencedirect.com/science/article/pii/S0896627321006772)
38. [Orbitofrontal Cortex as a Cognitive Map of Task Space](https://www.sciencedirect.com/science/article/pii/S0896627313010398)
39. [Population coding in the cerebellum: a machine learning perspective](https://pubmed.ncbi.nlm.nih.gov/33112717/)
40. [A bio-inspired bistable recurrent cell allows for long-lasting memory](https://arxiv.org/abs/2006.05252)
41. [Scale-Invariant Memory Representations Emerge from Moiré Interference between Grid Fields That Produce Theta Oscillations: A Computational Model](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6672484/)
42. [Scale-Invariance and Self-Similar 'Wavelet' Transforms: an Analysis of Natural Scenes and Mammalian Visual Systems](redwood.psych.cornell.edu/papers/field-1993.pdf)
43. [A Comprehensive Survey of Continual Learning: Theory, Method and Application](https://arxiv.org/pdf/2302.00487.pdf)
44. [Riemannian walk for incremental learning: Understanding forgetting and intransigence](https://arxiv.org/abs/1801.10112)
45. [Gradient Episodic Memory for Continual Learning](https://arxiv.org/abs/1706.08840)
46. [Signaling Mechanisms Linking Neuronal Activity to Gene Expression](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2728073/)
47. [Plasticity of the Nervous System & Neuronal Activity–Regulated Gene Transcription in Synapse Development and Cognitive Function](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3098681/)
48. [Neural Coding in Spiking Neural Networks: A Comparative Study for Robust Neuromorphic Systems](https://www.frontiersin.org/articles/10.3389/fnins.2021.638474/full)
49. [Lifelong learning of human actions with deep neural network self-organization](https://www.sciencedirect.com/science/article/pii/S0893608017302034)
50. [Brain-inspired learning in artificial neural networks: a review](https://arxiv.org/pdf/2305.11252.pdf)
51. [There's Plenty of Room Right Here: Biological Systems as Evolved, Overloaded, Multi-scale Machines](https://arxiv.org/abs/2212.10675)
52. [Universal Mechanical Polycomputation in Granular Matter](https://arxiv.org/pdf/2305.17872.pdf)
53. [A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II](https://sci2s.ugr.es/sites/default/files/files/Teaching/OtherPostGraduateCourses/Metaheuristicas/Deb_NSGAII.pdf)
54. [GraphVAE: Towards Generation of Small Graphs Using Variational Autoencoders](https://arxiv.org/pdf/1802.03480.pdf)
55. [Modeling by shortest data description](https://doi.org/10.1016/0005-1098(78)90005-5)
