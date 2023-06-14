# Learning to Learn: Evolving Generalized Self-Organizing Cyclic Spiking Computation Graphs
("Recurrent" sometimes has the connotation of only including edges from a node to itself, which is wholly insufficient for the feedback loops I'm picturing)

## Contents
- [Disclaimer](#disclaimer)
- [Contributing](#contributing)
  - [Guidelines](#guidelines)
- [Main Goals](#main-goals)
- [Why multiple learning paradigms?](#why-multiple-learning-paradigms)
- [Inspiration](#inspiration)
- [Proposed Solution Preview](#proposed-solution-preview)
- [Existing Literature](#existing-literature)
  - [Neuarl Architecture Search](#neural-architecture-search)
  - [Spiking Neural Networks](#spiking-neural-networks)
  - [Self-Modification / Self-Organization](#self-modification--self-organization)
  - [Meta-Learning / Multi-Modality](#meta-learning--multi-modality)
  - [Transformer Architecture](#transformer-architecture)
  - [In-Context Learning for Transformer-Based LLMs](#in-context-learning-for-transformer-based-llms)
  - [Spiking Graph Neural Networks](#spiking-graph-neural-networks)
  - [Universal Approximation](#universal-approximation)
  - [Compression as a Goal](#compression-as-a-goal)
  - [Continual Learning](#continual-learning)
- [Proposed Solution](#proposed-solution)
  - [MVP](#mvp)
  - [Final State](#final-state)
  - [Choosing Base Evolutionary Algorithm](#choosing-base-evolutionary-algorithm)
  - [Initial Population](#initial-population)
  - [Known Unsolved Problems](#known-unsolved-problems)
  - [Mechanisms that Might Evolve](#mechanisms-that-might-evolve)
- [Other Papers that Might be Useful](#other-papers-that-might-be-useful)
- [Future Ideas](#future-ideas)
- [Maybe Ideas](#maybe-ideas)
- [References](#references)

## Disclaimer
The following is an incomplete research proposal. It is not finished being designed and not all the mentioned papers have been finished yet (some haven’t even been started). Because of this, it is likely to change due to issues, contradictions, etc before implementation.

## Contributing
Due to limitations on my spare time, this is a public project that is open to feedback from anybody at any time. The idea is to provide a free non-degree credential for anybody looking to collaborate on and learn about an interesting research idea in neural architecture search by doing all learning in public view to simultaneously attack the stigma of making a mistake while also creating free online material from which future students can learn.

### Guidelines
- Use pull requests to suggest content modifications
- Definition of *content curator*: anybody whose content has been added to any branch
- Content modifications should be backed up by citations unless exempted by at least 75% of content curators
- Discussions space is open to anybody and should be used for debating content modifications in order to ensure adherence to the main tenets of the project (public learning as credentials and material for future students).
- Any content curator can call a vote to merge content at any time.
  - Voting closes a week after it is called or once success criteria is met
    - Success criteria: either at least 50% of all content curators or 50% of final votes after poll close are in favor.
  - Original author of content suggested in the discussions tab has first right to requesting the content be approved via pull request.

## Main Goals
- Improve model inference-time efficiency (less compute resources required along an arbitrary metric)
- Increase model representational density
- Incorporate multiple, complementary learning paradigms in a single network, i.e.
  - Supervised, unsupervised & reinforcement learning
  - Local and global
  - Different learning rates throughout the network (like how the hippocampus has to quickly form and hold on to episodic memories but the cortex learns more slowly)
  - Different state management methods
- Increase control of the tradeoff between memory and time complexity of final learner
- Mitigate the amount of human influence in the design of the model architecture and learning algorithm used in the final solution, instead focusing on easily compared empirical results
  - This will support all other goals

## Why multiple learning paradigms?
[How learning unfolds in the brain: toward an optimization view](https://www.sciencedirect.com/science/article/pii/S0896627321006772)

> “At the macroscopic level, studies have revealed that behavioral changes during learning are guided by different types of feedback such as supervision (Brainard and Doupe, 2002), reward (Schultz et al., 1997), and sensory prediction errors (Shadmehr and Holcomb, 1997), and develop on different timescales (Newell and Rosenbloom, 1981; Boyden et al., 2004; Smith et al., 2006; Yang and Lisberger, 2010).
> …
> More generally, it has been proposed that the brain learns new motor skills via supervised learning, unsupervised learning, and reinforcement—perhaps even simultaneously (Doya, 2000; Izawa and Shadmehr, 2011).”

## Inspiration
- Mitigating human influence in the learning paradigm and model architecture enough to support the other goals likely requires the use of an evolutionary approach
  - Unfortunately, whether the approach works for our purposes is also highly dependent on how it is implemented and what degrees of freedom it allows.
- Key fact: Biological neurons are able to modify their genetic expression based on environmental cues like when a specific neurotransmitter binds to a receptor in the cell wall (this can happen every time cells involved in a synapse communicate)
  - See [46](#references) and [47](#references)
  - In other words, they have many more degrees of freedom and a larger variety of types of DoF that they can utilize for learning than what we currently simulate
- Guiding Principle: Assuming a system can self-improve, increasing the diversity of computational bases on which the system can rely is likely to increase the probability that the system will be able to reach a better value for itself along a given dimension (efficiency, accuracy, etc.) by increasing the variety of options in the search space, which allows more possible combinations of tradeoffs at the cost of a likely increase in time spent searching (which will be mitigated later).
- The underlying computational mediums of the human brain and modern computers are different enough that we know there will be differences in the software that is optimal along some dimensions but we aren't sure exactly what or where those differences and dimensions will be plus predicting them is made even more difficult by the fact that we don't fully understand the brain and by the fact that modern digital computer hardware architecture is still changing and for certain uses, might even be supplanted by the use of quantum computing or neuromorphic hardware, which require new programming paradigms such as polycomputing[51](#references). 

## Proposed Solution Preview
- [Full Details](#mvp)
- Use an evolutionary algorithm to find a new learning algorithm / model architecture combination starting from initial population of existing algorithms / architectures
- Include flexible firing policy that can emulate multiple spiking mechanisms for activation sparsity and dynamic tradeoffs between spatial and temporal processing
- Allowing each node in the network to modify any aspects of any subgraphs in the network when it fires (including itself) also allows the model to dynamically change its own learning strategies
- Dynamic IO mapping & dynamic IO encoding
- Use constraints on task selection for each generation’s train / test datasets to encourage generalization by encouraging things like the use of multiple learning rates in a single network, which can be used to balance memorization & generalization since learning rate can be used as a regularizing force

## Existing Literature
### Neural Architecture Search
- Limitations of current methods
  - Limited adaptability: The final solution is static (even ones with continued online learning have a static learning paradigm)
  - Limited degrees of freedom: Most focus on topology and/or weights
     - “Such experimental trials convinced us that to solve the [Neural Architecture Search] dilemma, the connectionist paradigm alone is not adequate.” [1](#references)
  - "there is evidence that [weight sharing] inhibits the search for optimal architectures (Yu et al., 2020)" [5](#refernces)
  - Can take a while to find solution
     - [5](#references) addresses this by linearizing the loss function of untrained networks at the data points in each batch and using Kendall’s Tau correlation coefficient, which measures the strength and direction of association that exists between two variables, to determine that an initialized network is not worth training. It is uncertain whether this generalizes to harder tasks. It also remains to be seen whether it can be used for an unsupervised context. It might be adaptable to reinforcement learning but that has not been shown and likely would prove unstable due to the general sparsity of signal in the reinforcement context. It's also unclear if this method can be adapted to spikiing neural networks.

### Spiking Neural Networks
- There are important advantages to being able to utilize the temporal dimension for encoding information
  - Allows easy local update rules for training such as Spike-timing-dependent plasticity (STDP), which “is the primary candidate mechanism for the storage of information in autoassociative networks” [12](#references) in the brain
  - When used in conjunction with neuromorphic hardware, they can provide a large reduction to energy consumption by switching to an event-driven model of computation
  - Including time as an explicit degree of freedom in the model allows it to more closely approximate the actual dynamics of the world.
  - "the ability to process noisy and dynamic data" [50](#references)
  - "more robust and fault-tolerant computing" [50](#references)
- “Spiking Neural Networks have revealed themselves as one of the most successful approaches to model the behavior and learning potential of the brain, and exploit them to undertake practical online learning tasks” [6](#references)
- Main existing training methods: surrogate gradients, spike-op gradient descent, realtime recurrent learning, forward gradient propagation, event-based backpropagation
  - The biggest drawback of most of these is a decrease in training efficiency in some way or (in the case of converting non-spiking to spiking) an increase in inference time but the accuracy has been shown to be comparable to the similarly trained non-spiking version of the network. This further means that the network could dynamically switch between these two modes to change time & energy tradeoffs insitu based on task necessity if the proper conditions are met for the evolution of such a mechanism.

### Self-modification / Self-organization
- In [14](#references) they "consider a dynamic network to be a network where nodes and edges appear and/or disappear over time”. This is merely a less powerful subset of what I am proposing. They also state that self-modification allows the model to “radically influence network properties which enables a more powerful representation of network data which in turn increases predictive capabilities of methods using such data.”
- In [49](#references) they attempt to overcome the shortcomings of a previous experiment that used a hand-crafted data set, which is "against the idea of ... learn[ing] significant features by iteratively tuning internal representations" and the increased "dimensionality of neural weights along the hierarchy" caused by preprocessing.
  - "The most well-established model is the self-organizing feature map (SOM, Kohonen, 1990) algorithm that nonlinearly projects a high-dimensional input space onto a low-dimensional (typically two-dimensional) discretized representation. It consists of a layer with competitive neurons connected to adjacent neurons by a neighborhood relation."
    - Fixed topology
    - fixed number of neurons
  - "The ability of a network to create new neurons (and remove unused ones) for adapting to novel incoming signals is crucial for better dealing with non-stationary input distributions."
  - "A well-known model is the growing neural gas (GNG,  Fritzke, 1995), in which neurons are added at fixed intervals to minimize local errors. In contrast to the GNG, the growing when required (GWR) network proposed by Marsland et al. (2002) creates new neurons whenever the activity of trained neurons is smaller than a given threshold. ... Additionally, the algorithm considers the number of times that neurons have fired so that recently created neurons are properly trained before creating new ones."
  - "The use of an activation threshold and firing counters to modulate the growth of the network leads to the creation of a larger number of neurons at early stages of the training and then tune the weights of existing neurons through subsequent training epochs. This behavior is particularly convenient for incremental learning scenarios since neurons will be created to promptly distribute their receptive fields in the input space, thereby yielding fast convergence through iterative fine-tuning of the topological map. It has been shown that GWR-based learning is particularly suitable for novelty detection and cumulative learning in robot scenarios Marsland et al. (2005), Marsland et al. (2002)."
    - In GNG, number of neurons grow linearly over time
    - In GWR, number of neurons grows quickly in beginning but has a horizontal asymptote as training progresses
    - GWR tends to have quicker convergence with respect to "quantization error (average discrepancy between the input and representative neurons in the network)"
    - "The standard GNG and GWR learning algorithms do not account for temporal sequence processing. Consequently, these models were extended with recurrent connectivity"
  - Recurrent self-organization
    - "in the [temporal kohonen map], there is no explicit back-reference to previous map activity because the context is only implicitly represented by the weights."
    - "In Parisi, Magg, and Wermter (2016), we presented a GWR network equipped with recurrent neurons with one context descriptor, yielding a decreased temporal quantization error for a time-series prediction task with respect to recurrent models of the SOM and the GNG."
  - Lifelong learning
    - "The functional plasticity of the map can be modulated through critical periods, i.e., a particularly sensitive period in which the map is more responsive to the effects of experience."
    - " In the SOM (Kohonen, 1990), two training phases are used to achieve a topographic organization resembling the two phases of a critical period: the organization phase in which the network is trained with a high learning rate and large neighborhood size, and the tuning phase in which these parameters are reduced to fine-tune the map and learn more detailed distinctions of the input."
      - Fixed number of neurons forces critical periods to play a "crucial role in the formation of maps with good topological organization"
      - When not fixed, new resources can be created based on input distribution
        - Hyperparameters kept fixed and uses competitive Hebbian learning so no critical periods
        - Tracking the number of iterations since connection use can allow control of information durability by deleting connections that haven't been used and nodes without connections on each iteration
    - Predictive coding can be used for lifelong learning
  - Proposed method: hierarchical self-organization

### Meta-learning / Multi-modality
- In the meta-learning paradigm, multiple tasks are frequently used in conjunction in order to find a good starting point for a given architecture across multiple different tasks, which are later fine tuned for specific tasks.
- [18](#references) presents an approach that finds a good common network initialization for learning based on interpolation of losses from multiple tasks. Doing so usually leads to an improvement in generalization. This approach will use that as a subset of its search space while also including initial model architecture, loss function, optimization algorithm, etc.
- Including multiple modalities in the learned tasks improves the robustness and generalizability of the learned representations
- In [19](#references) they mainly focus on tokenization scheme and scaling the transformer architecture to log linear performance over increasing input & output sizes.
- In [20](#references) they develop a similar tokenization scheme for each modality used that is fed to a transformer model in order to allow for generalization across modalities for static input nodes but their task list is not well curated for positive transfer and likely still separates the tasks to a degree, which is supported by appendix J showing some amount of negative transfer.
- In [21](#references), pulling from earlier work (PaLM: Scaling Language Modeling with Pathways) they change the multi-head attention in a transformer to a multi-query attention mechanism that allows the key and value projections to be shared, which allows for faster autoregressive decoding at inference time. They also 


### Transformer Architecture
- Main advantage is scalability (in terms of parallelization) of the attention mechanism, which allows for information routing, followed by a traditional MLP. This allows the learning of long-range dependencies at the expense of a fixed-length input size for the sequence. The parallel processing of input tokens often adds an additional requirement to i.e. include the position in each input token’s embedding.
- “Another inherent limitation of Transformers on sequential tasks is the lack of recursive computation (Dehghani et al., 2018), and the number of transformations possible on the input is bounded by the model depth. Such disadvantages have impact on tasks that require careful tracking of a world state or modeling hierarchical structures (Tran et al., 2018; Hahn, 2020). On the other hand, while RNNs can maintain an internal state for an unbounded time while accumulating more computations upon it, the size of this internal state is limited by the dimension of the hidden state.” [26](#references)
- “the current hidden representation of a Transformer only accesses the past representations of lower layers, even though higher level representations of the past have already been computed as an autoregressive model.” [26](#references)
- Supposedly reducible to a combinatorial Hopf Algebra [25](#references)
  - “The representation theory of a Hopf algebra is particularly nice, since the existence of compatible comultiplication, counit, and antipode allows for the construction of tensor products of representations, trivial representations, and dual representations.” - Wikipedia

### In-context Learning for Transformer-based LLMs
- In-Context Learning does not actually modify the underlying network in any way (no actual learning occurs)
- ICL in smaller (tractable) attention-only transformer-based LLMs likely relies on skip trigram detectors and induction heads that act as “in-context nearest neighbors” [Chris Olah]
  - Is also a subset of what I’m proposing
  - Could be other mechanisms involved when adding the MLP but they would also be evolvable by my method if they provide a benefit
- Not able to leverage deduction very well

### Spiking Graph Neural Networks
- “The striking feature of this plot is its self-similarity - in some ways it looks like a fractal. … figure 5 strongly supports the claim that the degree distribution of a graph received from this numerical experiment follows a power law (similar plot was obtained in a number of simulations).” [30](#references)
  - This type of scale-invariant, self-similar dynamics could also be exploited by my approach assuming the use of a continuous time implementation. I’m unsure about the discrete case but my intuition says that it should work there as well to some extent. This is actually something I expect will eventually happen (possibly with power law exponent changes determining emphasis on local vs global connectivity). I expect this type of dynamics is a convergent strategy for efficiently tiling an arbitrarily contracting / expanding space.
    - “If the inhibitory tuning is smoother than the excitatory tuning (Figure 1b), the output neuron develops equidistant firing fields, reminiscent of grid cells on a linear track (Hafting et al., 2008). If instead the excitatory tuning is smoother, the output neuron fires close to the target rate of 1 Hz everywhere (Figure 1c); it develops a spatial invariance. For spatially untuned inhibitory afferents (Grienberger et al., 2017), the output neuron develops a single firing field, reminiscent of a one-dimensional place cell (Figure 1d); (cf. Clopath et al., 2016).” [31](#references)

### Universal Approximation
- [32](#references) shows that single hidden layer networks with bounded width are still universal approximators for univariate functions, but this property is no longer true for multivariable functions.
- According to Wikipedia, “The works of Vladimir Arnold and Andrey Kolmogorov established that if f is a multivariate continuous function, then f can be written as a finite composition of continuous functions of a single variable and the binary operation of addition. [4] … In a sense, they showed that the only true multivariate function is the sum, since every other function can be written using univariate functions and summing. [6]” (reference [4] = http://www.math.toronto.edu/drorbn/Talks/Fields-0911/; reference [6] = https://statweb.stanford.edu/~cgates/PERSI/papers/nonlin_func.pdf)
- These works taken together with the fact that my proposed network finding framework has the ability to revert to the more simplistic dynamics of learning a feedforward network with a single hidden layer but also has a non-fixed functional class for activation (it can use any function for a node’s activation value) and no constraints on the architecture of any hidden layer means that my framework can evolve a learner capable of finding a universal approximation of any desired function using a finite number of nodes.

### Compression as a Goal
- My understanding is that universal approximation in a finite number of nodes is easy to obtain in a computational base. Thus, we turn to determining the amount of maximal compression that a self-improving system with the goal of compressing its own dynamics can achieve.
- [33](#references) explores "the notion of compression as a means of extending the temporal receptive field of Transformer-based sequence models. … We see the idea of compressive memories is applicable not only to the modality of text, but also audio, in the form of modelling the waveform of speech, and vision, within a reinforcement-learning agent trained on a maze-like memory task. … The main limitation of this work is additional complexity, if the task one wishes to solve does not contain long-range reasoning then the Compressive Transformer is unlikely to provide additional benefit. However as a means of scaling memory and attention, we do think compression is a simpler approach to dynamic or sparse attention — which often requires custom kernels to make efficient.” This is only relevant to provide a small amount of evidence that allowing the model to compress its input is a good idea. This is to be accomplished by allowing the model to dynamically determine it's own input mapping and number of input nodes.
- Perceptual quality can be defined as “the extent to which [an output sample] is perceived as a valid (natural) sample”. In [35](#references), they adopt a mathematical definition of this that was proposed by others and show that “there is a triple tradeoff between rate, distortion and perception.” “Rate” here means bits per sample and “distortion” means the distributional shifts in the data. Their "key observation is that the rate-distortion function elevates as the perceptual quality is enforced to be higher (see Fig. 1). In other words, to obtain good perceptual quality, it is necessary to make a sacrifice in either the distortion or the rate of the algorithm. … But our theory shows that every distortion measure (excluding pathological cases) must have a tradeoff with perceptual quality.” They also note that even though the divergence function that best relates to human perception is the subject of ongoing research, their results “hold for (nearly) any divergence.” This effect is particularly pronounced at low bit rates.
- Compression being the main goal of the paradigm presented in [36](#references) for encouraging generalization fits very well with my approach as a major goal of my approach is to encode the desired behavior in a highly dynamical system, allowing for higher representational density (a.k.a. compressing the function)

### Continual Learning
According to [43](#references), blurred boundary continual learning is when the "task boundaries are blurred, characterized by distinct but overlapping data label spaces". This is the most relevant continual learning setting I have found but most continual learning settings apply here given the breadth of the main goals. There are two main metrics for memory stability, forgetting measure (FM) [44](#references) and backward transfer (BWT) [45](#references). FM "is calculated by the difference between its maximum performance obtained in the past and its current performance" [43](#references). "BWT evaluates the average influence of learning the k-th task on all old tasks" [43](#references). In direct contrast, plasticity can be measured using intransience (IM) [44](#references) and forward transfer (FWT) [45](#references). "IM is defined as the inability of a model to learn new tasks, which is calculated by the difference of a task between its joint training performance and continual learning performance" [43](#references). Memory stability and plasticity have an intrinsic tradeoff. One way to deal with this is by including the training of a generative model to use for replay-based methods at the cost of additional overhead but generative models are also plagued by the catastrophic forgetting problem. Alternatively, a Bayesian framework can be used but the posterior probability has to be estimated due to intractability. The main estimation methods for the posterior probability are Laplace approximation and variational inference. "The constraint on continual learning for either replay or regularization is ultimately reflected in gradient directions" so some methods attempt to manipulate the gradient-based optimization in order to handle the plasticity-stability tradeoff but these methods are mainly based on finding a shared solution for all incremental tasks, which is subject to severe inter-task interference [43](#references). Another class of methods to incrementally learn new tasks called architecture-based approaches tries to do so in a partially separated way by specifying which parameters are shared and which are task-specific [43](#references).

## Proposed Solution
### MVP
- Discrete time
- Borrowing from spiking neural networks, we include a firing policy for every node that determines whether or not it should fire given it’s activation value and any internal state the policy maintains
  - Firing policy flexibility (always fire, threshold, decay, random, etc) allows a dimension for tradeoff between spatial (memory-based) and temporal processing
    - See section “1. Neural variability shapes learning, but is often inflexible” in [37](#references) for justification of including random firing policy. In short, it is because behavioral variability during learning is a key part of exploring the reward space.
- Model can determine how each input & output will be encoded w/ choice of rate coding, temporal coding, phase coding, and burst coding [48](#references)
  - Different bits of information in the network will have different transmission requirements (required noise tolerance, relative transmission speed, etc)
- When switching tasks, feed in the new task embedding so the model can reconfigure the IO mapping and the number of IO nodes, internal context, etc.
- Allow nodes in the network, upon firing, to each apply some unique ordered list of effects to a subset of their outgoing connections
  - Each node also has a set of policies for choosing which of its outgoing edges to use when applying its effects (many-to-many mapping)
    - Examples policies: newest; oldest; most or least used; most or least recently chosen in general or by x effect. Will be wasteful to track all of the necessary variables here for every node so consider each node having a mutable set that determines which pointers to track.
  - A node can have effects that require different numbers of connections
  - Each node will also have something similar to a refractory period represented as a number of touches before being able to forward its activation value to outgoing connections
  - Effects also have a connection hop policy to determine how far the effect should apply, mimicking locality of neuromodulator effects in the brain
  - Possible effects:
    - Modify a numerical value (add, subtract, multiply, divide, set, exp, etc.)
    - Modify a categorical value (even something like the operation being performed by an effect)
    - Modify a policy (class type or properties of each policy like threshold or decay)
    - Change a node’s effects (order, replace, modify individual ones, etc.)
    - Create or destroy edges / nodes / effects
      - Use increasing max numbers of nodes / edges / effects that slightly increase each generation to hopefully spend less computation on a less viable strategy
    - Change input / output node mapping to data / meaning
      - Number of necessary IO nodes is different per task
      - Allows for tradeoff between time and memory costs in scalability of input / output sizes (like a saccade)
- Use an evolutionary algorithm to find the final starting point for the learning algorithm / model combination
  - Incorporate time cost and memory cost into the fitness function
    - e.g. for each task, a\*score + b/normalized_mem_cost + c/normalized_time_cost, where a+b+c=1 w/ d weight on area under this curve when plotted over each training iteration and e weight given to the value over the test set, where d+e=1 and take the "macro" mean (equal weight to every task).
      - Score would be something like F1 or MCC divided by (false_positives+1) for supervised classification tasks (or things like BLEU for translation) and the normalized total reward for reinforcement context. Basically each task has to define its own metric.
        - Dividing the combined score by the number of false positives in the supervised classification context should heavily bias the evolved learner toward deduction without forcing it to only use deduction. My intuition says false positives are generally more costly than false negatives so start with that and then compare to dividing by false negatives.
      - This adds competing evolutionary pressures. One to generalize to unseen data (test set fitness) and another to quickly learn or memorize (area under training curve) with the ability to change each factor’s relative importance
    - This also adds pressure to compress the behavioral dynamics across time and memory
  - Start with an initial population that is implementations of existing algorithms in this framework ([details](#initial-population))
    - Might be able to get a head start on this by generating computation graphs from open source code
    - Use a boolean on each origin species that prevents it from full collapse for the first X generations (minimum population of Y) to ensure the final learner has a fair chance of incorporating the important components from the initial population.
      - We won’t know when adding another piece of the initial population to a species would be beneficial.
- To allow for reinforcement, supervised and unsupervised learning in the same network, there will be at least five special input nodes (likely more). One will represent reward for reinforcement learning. For each output node, include an extra input node for that node’s loss (minimum of one node). The remaining three in the minimum set are to signify which of these learning types should be used.
- Set aside a held out data set (equal distribution across every task) to ensure generalization of final learner
- To avoid obscure failure modes caused by data quality issues or poisoning attacks, use smaller hand-crafted datasets that are designed to teach specific lessons and, when possible, use data generators that are designed to teach specific lessons instead. -> still vulnerable to reward hacking w/ data generators?
- Basic reinforcement learning tasks necessary for a cognitive map of task space [38](#references)
  - Data can easily be generated on the spot for most of these because the modality and dimensions don't particularly matter since we can define the task to just choose a random modality, input dimensions (above a different min per task), target action embeddings, target stimulus (might want to add Gausian noise), etc.
  - Probabilistic and non-probabilistic reversal learning
  - Delayed alternation
  - Extinction training should help lead to creation “of a new competing association” for cases in which a preferred outcome is no longer available
    - In order to encourage desired behavior, must include same task in test set but with different outcome metrics (spontaneous recovery, reinstatement, rapid reacquisition or renewal)
  - Devaluation should encourage “a capacity to ‘simulate’ the consequences of actions within a cognitive model of the task” based on results comparing their sham-lesioned model to their OFC-lesioned model
    - The authors also claim that a similar line of reasoning can be used to explain the role of OFC in model-based RL such as sensory preconditioning, identity unblocking and pavlovian overexpectation.
  - "Recent results implicate medial OFC in encoding economic value and lateral OFC in more complex functions, such as credit assignment and model-based RL (Noonan et al., 2010, Rudebeck and Murray, 2011a, Rudebeck and Murray, 2011b, Noonan et al., 2012)."
- Task Selection
  - Use a skill-tree based approach where harder tasks cannot be unlocked until all pre-reqs have been completed w/ ≥ X score by at least N models.
  - In every generation, the models will use freshly chosen training and testing datasets that will be randomly sampled ordered subsets of examples from the available tasks. Training for each model of that generation occurs for the same random number (between hyperparameters A and B) of iterations over the subset before being tested.
    - Each generation, a new random number of iterations is chosen
      - The inclusion of randomization in determining the number of training iterations will at different times encourage either fast learning or slow learning (putting more or less weight on learning aspects of the current example respectively, remembering that learning rate can be used as a regularizing force) so that there is less pressure toward a specific uniform “learning rate” across all species. This will hopefully create an opportunity for them to be combined into a single network, noting that speciation (see NEAT) should help prevent the premature collapse of the circumstances that offer this opportunity but using speciation for this introduces the need for an additional mechanism that allows for breeding across species.
        - Having a model with multiple learning rates could additionally allow for spontaneous recovery of memory even after “extinction training” [39](#references) and thus improve the chances of extinction training encouraging the creation ‘of a new competing association’ for cases in which a preferred outcome is no longer available
      - Later generations will have larger datasets, allowing to spend more time on evaluating the differences between models that are likely better performing (controlled by hyperparameter: % or absolute dataset examples increase per generation)
        - This plus random interleaving of the tasks and slightly increasing max numbers of nodes / edges / effects each generation will hopefully help mitigate catastrophic forgetting without forcing the early population to deal as heavily with the issue due to having fewer tasks to learn.
- Task embeddings
  - Supervision method
  - Available data types
    - Must allow for unspecified data type (all 0s)
    - Dimensions of each data type
  - Document embedding of task description in natural language
    - Might be better to allow it to "define its own embedding" by using the dynamic input mapping

### Final State
- Continuous Time - **Is it worth it? How can the modulatory effects be incorporated into the differential equations and the solver?**
- Models are able to determine their own training set of a given size (task only, not example) but are not allowed to choose any of the tasks in the evaluation metric for the current generation or any of the locked tasks
  - Task choice should be based on task embedding (ask the model to spit out a task embedding it’s looking for and find most similar task, give random example)
  - Forcing different tasks in the test set will put extra pressure on the model to determine how to best generalize and be sample efficient
  - Will require adding an additional input node to say what task to choose next as well as multiple output nodes to determine task embedding

### Choosing Base Evolutionary Algorithm
- NEAT [4](#references)
  - Remove the bias toward smaller structures since it is supplanted by the use of the inverse normalized time and memory costs in the fitness function.

### Initial Population
#### SNNs
  - Delay Learning methods:
    - Delay selection
    - Delay shift
  - Learning single spike per neuron: deSNN
    - “The model works on the order of input spikes and it does not have a leakage in the generated PSP. The leakage can be an essential property of a biological neuron that is sensitive to the time interval and consequently it can be sensitive to temporal information inside the spatiotemporal input pattern. Although, each synaptic input in deSNN can have multiple spikes, the output can generate a single spike.” [8](#references)
  - Attention Mechanism: Multi-dimensional Attention
    - “To adapt attention SNNs to a variety of application scenarios, we merge multidimensional attention with SNN (MA-SNN), including temporal, channel, and spatial dimensions, to learn ’when’, ’what’ and ’where’ to attend, respectively. These attention dimensions are exploited separately or simultaneously according to specific task metric requirements such as latency, accuracy, and energy cost. Classic convolutional block attention module (CBAM) [28] is adopted as the basic module to construct MA-SNN.” [13](#references)
  - Learning multiple spikes in single neuron or single layer:
    - “Chronotron has two versions: I-learning with high biological plausibility and E-learning with memory capacity and high computational cost.” [8](#references)
    - Synaptic weight association training (SWAT) algorithm
      - Trains neurons in a single layer
      - synapse is depressed or potentiated depending on firing rate threshold for post-synaptic node [8](#references)
      - “In SWAT, BCM[ (Bienenstock–Cooper–Munro learning)] is used to modulate the height of an STDP learning window to stabilise the weight adjustment governed by STDP. While STDP and BCM are used to train the output layer, the hidden layer in SWAT is used as a frequency filter to extract features from input patterns. The method only can use rate coding in the input and output patterns.” [8](#references)
    - DL-ReSuMe (A Delay Learning-Based Remote Supervised Method for Spiking Neurons) (Taherkhani, Belatreche, Li, & Maguire, 2015b)
      - “integrates weight modulation with the synaptic delay shift to map a random spatiotemporal input spike pattern into a random desired output spike train in a supervised way.” [8](#references)
      - “can learn input spike trains of shorter duration and smaller mean frequency with a higher accuracy and much faster learning speed than ReSuMe. One interesting feature of DL-ReSuMe method is the ability to solve the silent window problem in a spatiotemporal input pattern.” [8](#references)
    - Liquid State Machine (LSM) allows the use of an existing reservoir of recurrent connections to act as short term memory
    - unsupervised synaptic plasticity driven by STDP
#### Transformers
#### LSTMs
  - [40](#references)
#### CNNs
#### GNNs
#### HTMs
#### Reinforcement learning algorithms
#### Optimization methods
  - ADAM, gradient descent, forward-forward, etc.
#### Loss functions
#### Traditional methods (This is a computation graph so it can include anything)
#### Non-learning algorithms (not part of population but can be referenced by genes)

### Known Unsolved Problems
- Ensuring alignment is maintained in production without reverting to the original network in between tasks so that online learning can take place. RLHF might not be a scalable enough solution and RLAIF could be too dangerous.
- How to efficiently parallelize both the evolutionary algorithm using federated learning and the final result
- How to incorporate continuous time with the modulatory effects
- How to encourage the use of unsupervised tasks for learning
- How to curate the available tasks in a way that will encourage generalization to new tasks and modalities
- Is there a way to include unsupervised tasks in the eval set, especially in an unbiased way (clustering method metrics like silhouette or penalizing intravariance introduce intrinsic bias in the signal)? Closest proxy I can think of is self-supervised like an LM or autoencoder but even those have bias in the loss function itself and the whole point of including unsupervised tasks is to allow it to learn in an unbiased manner from ambient stimuli.
  - Could we induce the networks to incorporate unsupervised learning after a number of generations by including unsupervised learning for each modality before the training for that generation?
  - Without this, might not be worth having unsupervised learning
- Whether I will need to incorporate compression into the main goal or if the time and memory complexity is a good enough proxy
- How to best allow the mixing of model architectures and optimization methods
  - Maybe try to just equally disperse the optimization methods used by each initial agent within a species. In addition, could force each piece (model architecture, optimization algorithm, loss/reward function, etc.) to compete ONLY with other options of the same type and score them all separately.
  - Also could record the performance of each model along each dimension of the fitness function and then choose mating partners based on complementary performance metrics but this still leaves it wondering how to optimally mix the parents.
- Translation between genes and network

### Mechanisms that Might Evolve
- Stabilization of the system away from an infinite positive feedback loop between two nodes or two populations
  - Removal of nodes / edges in the case of prolonged overstimulation (maybe using a critical-point-based phase transition like in percolation for detection)
    - seems unlikely that the node would be deleted though due to the current relative density of information contained there vs the edges. Perhaps I should allow dynamically adding edge parameters for storage to even out the parameter density?
- Reordering mechanism that increases representational capacity of self-organizing effects
  - Modularizing the effects into groups that are interspersed with reordering effects
  - Similar to the splicing mechanisms that evolved in DNA that allow for multiple proteins to be encoded by the same stretch of DNA
- Scale-free dynamics induced by the need to be able to tile an arbitrary space that is potentially expanding or contracting at different times
  - Similar to what grid cells & place cells do (see [41](#references) and [31](#references))
  - Also, see [42](#references)

## Other Papers that Might be Useful
- [On the Relationship Between Variational Inference and Auto-Associative Memory](https://arxiv.org/pdf/2210.08013.pdf)
  - "In order to improve the memory capacity, modern Hopfield networks [22, 21, 8] propose several variants of the energy function using polynomial or exponential interactions. Extending these models to the continuous case, [30] proposed the Modern Continuous Hopfield Network (MCHN) with update rules implementing self attention, that they relate to the transformer model [36]. In [26], the authors introduce a general Hopfield network framework where the update rules are built using three components: a similarity function, a separation function, and a projection function."
  - "It has been shown that overparameterized auto-encoders also implement AM [28, 33]. These methods embed the stored patterns as attractors through training, and retrieval is performed by iterating over the auto-encoding loop."
- [Multifactorial Evolutionary Algorithm with Online Transfer Parameter Estimation: MFEA-II](https://www.researchgate.net/profile/Abhishek-Gupta-17/publication/331729696_Multifactorial_Evolutionary_Algorithm_With_Online_Transfer_Parameter_Estimation_MFEA-II/links/5c98495892851cf0ae95ec75/Multifactorial-Evolutionary-Algorithm-With-Online-Transfer-Parameter-Estimation-MFEA-II.pdf)
- [STCA: Spatio-Temporal Credit Assignment with Delayed Feedback in Deep Spiking Neural Networks](https://www.ijcai.org/proceedings/2019/0189.pdf)
- [Sparse Distributed Memory using N-of-M Codes](apt.cs.manchester.ac.uk/ftp/pub/apt/papers/NofMnnV3.pdf)
- [A discrete time neural network model with spiking neurons: II: Dynamics with noise](https://arxiv.org/abs/1709.06206)
- [The Information Bottleneck Problem and Its Applications in Machine Learning](https://arxiv.org/pdf/2004.14941.pdf)
  - “The information bottleneck (IB) theory recently emerged as a bold information-theoretic paradigm for analyzing DL systems. Adopting mutual information as the figure of merit, it suggests that the best representation T should be maximally informative about Y while minimizing the mutual information with X.” (a.k.a. compression)
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
  - "Although the general principles that govern embedding dimensionality are not known, several computationally inspired hypotheses have been proposed. One dominant hypothesis is that the information in any given brain area is extracted by other areas and peripheral systems through a linear readout. A linear readout scheme is a weighted average of the activity across neurons and has an intuitive geometric interpretation: it is a projection of the neural activity in the full Euclidean state space along a specific direction. One common strategy to evaluate this hypothesis is to quantify the degree to which a linear decoder can extract desired information from population activity in a brain area."
  - "An extension of this hypothesis is that embedding across a population of neurons is organized such that different linear decoders can extract information about different task-relevant variables without interference. For example, movement and movement preparation signals in monkeys’ motor cortex reside in orthogonal subspaces [53]. This organization could ensure that preparation does not trigger movement [54,55], and may help protect previous motor memories [56]. Similar principles might govern interactions across cortical areas, by confining to orthogonal subspaces information that is private to an area and information that is communicated [57,58] ... We note however, that despite strong enthusiasm, direct evidence that the brain processes and/or communicates information through orthogonal linear decoders is wanting."
  - "In general, high-dimensional embeddings with more degrees of freedom can facilitate extraction of task-relevant variables without interference (Rigotti et al. 2013; Cayco-Gajic et al. 2017; Cayco-Gajic and Silver 2019; Litwin-Kumar et al. 2017; Lanore et al. 2021). However, embedding information in arbitrarily high dimensional subspaces can have adverse effects for generalization [64]. To improve generalization, embeddings have to be appropriately constrained to capture the structural relationships and inherent invariances in the environment [10,38,65]. A theory for the ensuing trade-offs has been recently developed for the case where the activity is organized in multiple disjoint manifolds corresponding to different categories [66] and applied to interpret representations in the visual system [67] and in deep networks [68]. It is also possible to organize information embedding such that certain linear projection reflect information in more abstract form and therefore enable generalization, while others enable finer discrimination [15]."
  - "The utility of linear decodability however, becomes less clear for intermediate stages of information processing in higher brain areas that carry information about latent variables that support flexible mental computations. While an experimenter may apply linear decoders to find information about a hypothesized latent variable in a certain brain area, there is no a priori reason to assume that the brain relies on such decoders."
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

## Future Ideas
- Using federated learning across a variety of different hardware implementations might be usable to improve generalization across different types of hardware
- Interpolation or mating between each agent before and after its training to allow some computation reuse without letting too much leakage of a previously trained-on example being included in a future generation’s test set
  - possibly using something like [Git Re-Basin](https://arxiv.org/abs/2209.04836) but use the before model instead of a differently initialized one plus put more weight on before than after training
    - Also would have to figure out how to deal with node, edge and effect creation / deletion
    - Would have to use some modification of the activation matching algorithm in order to deal with spiking (not sure how to incorporate with the straight-through estimator)
- Determine “success rate” of cross-species and of cross-time mating strategies and put more likelihood on using more successful strategies
- Try to use roughly equal distribution over task context (supervised, reinforcement, etc) to avoid giving an intrinsic advantage to some members of the initial population

## Maybe Ideas
- Tasks should be paired with each task (not example) requiring a generalization of others so that the generalized task can be selected for the eval set in a way that encourages generalization (i.e. a task that requires visual and audio would be a generalization of tasks that include audio and of tasks that include visual if and only if the underlying audio and visual tasks are required in order to complete the final task)
- Each time a task is selected for the training set, include a generalized task in the eval set
- Meta optimize this algorithm using itself like NVIDIA did with its simulator?
- Effect diffusion (static or dynamic)
  - Like RNA inside exosomes not neuromodulation
- Augment the input data of an example every successive time it is used in order to help prevent memorization - would need to find a modality / task / etc - independent method that won’t change the semantic meaning of the input data or would have to specialize the augmentation per modality or task

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
