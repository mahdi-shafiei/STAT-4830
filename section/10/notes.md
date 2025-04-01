---
layout: course_page
title: Benchmarking Optimizers - Challenges and Some Empirical Results
---

# Benchmarking Optimizers: Challenges and Some Empirical Results

[Cheat Sheet](cheatsheet.md)

## Table of contents
1. [Introduction: The Optimizer Comparison Problem](#introduction-the-optimizer-comparison-problem)
2. [Challenge 1: Defining and Measuring Speed](#challenge-1-defining-and-measuring-speed)
3. [Challenge 2: The Hyperparameter Tuning Trap](#challenge-2-the-hyperparameter-tuning-trap)
4. [A Solution: Rigorous Benchmarking Principles](#a-solution-rigorous-benchmarking-principles)
5. [What We Learned: AlgoPerf Competition Results](#what-we-learned-algoperf-competition-results)
6. [Conclusion](#conclusion)

## Introduction: The Optimizer Comparison Problem

In previous lectures, we studied several optimization algorithms commonly used in machine learning. We started with Stochastic Gradient Descent (SGD) for a simple mean estimation problem in Lecture [6](../6/notes.md). We then analyzed SGD, Momentum, Exponential Moving Average (EMA), and Preconditioning using the Noisy Quadratic Model (NQM) in Lecture [7](../7/notes.md). Finally, Lecture [9](../9/notes.md) introduced adaptive methods like Adagrad, Adam, and AdamW. These lectures studied training methods on simplified problems. But how do these optimizers compare in practice when training complex deep learning models?

Determining which optimizer is "better" or "faster" for deep learning tasks is difficult. Simple approaches, like comparing final loss values or visually inspecting training loss curves, often provide misleading conclusions. The reasons for this difficulty are multifaceted and involve how we define speed, how algorithms interact with specific problems, and how hyperparameters are selected.

Our theoretical analyses hinted at some complexities. We saw how SGD with a constant learning rate converges to a region around the optimum due to gradient noise (Lecture 6). The NQM highlighted how problem conditioning (the ratio of largest to smallest curvature, $\kappa = h_1/h_2$) affects convergence rates and how different methods address this (Lecture 7). While these theoretical models are instructive, they do not capture the full complexity of comparing optimizers on large-scale, non-convex deep learning problems.

This lecture explores the challenges of empirically comparing optimizers. We will first examine the difficulty in precisely defining and measuring optimization speed. Second, we will investigate the crucial role of hyperparameter tuning protocols, showing how they can drastically alter comparison outcomes. Third, we will introduce the principles of rigorous benchmarking as a solution, using the AlgoPerf benchmark (Dahl et al., 2023) as a case study. Finally, we will discuss concrete results from recent benchmark evaluations (Choi et al., 2019b; Kasimbeg et al., 2024) to understand the current landscape of optimizer performance.

## Challenge 1: Defining and Measuring Speed

A fundamental challenge in comparing optimizers is defining what it means for one to be "faster" than another. A common approach is to plot a performance metric, like validation loss, against training time or steps for different optimizers. However, these curves often intersect, sometimes multiple times during training. One optimizer might show faster initial improvement, while another might achieve a better final result or converge more quickly later in training. This makes it difficult to declare a definitive winner based solely on the shape of the training curves.

Figure 1 from Dahl et al. (2023) illustrates this problem. The plot on the left shows validation error curves from two distinct training runs using the same optimizer and task. The curves clearly cross, making it ambiguous which run is "faster" overall. The plot on the right shows the best validation error achieved up to that point in training (the running minimum) for the same two runs. Even using this metric, which smooths out some noise, the curves still intersect, indicated by the red crosses. This shows that determining which run performs better depends on when you stop training.

> **Figure 1 (from Dahl et al., 2023):**
> ![Optimizer Comparison Curves](figures/dahl_fig1.png)
> *Caption Quote:* "Figure 1: A direct comparison of training curves is ill-posed if they intersect. Left: The validation error for two different runs... Right: The best validation error obtained so far by each curve... the curves intersect multiple times (highlighted by red crosses (✖) at the bottom of each plot)." (Dahl et al., 2023, p. 10)

To overcome this ambiguity, a standard approach is to measure *Time-to-Result*. This involves pre-defining a specific *target* performance value for a relevant evaluation metric (e.g., target validation accuracy or test error rate). The performance of an optimizer on a given workload is then measured by the time (or number of steps, under controlled conditions) required to *first reach this target value*. This method provides a single, unambiguous number for comparison, assuming the target is achievable by the optimizers being compared (Dahl et al., 2023, Sec 4.1). It shifts the focus from the entire trajectory to a specific, meaningful milestone.

## Challenge 2: The Hyperparameter Tuning Trap

Beyond defining speed, perhaps the most significant challenge in comparing optimizers is handling*hyperparameter tuning. Optimizers like SGD, Adam, or Momentum are not fully specified algorithms but rather templates that require setting hyperparameters, such as learning rate, momentum coefficients ($\beta_1, \beta_2$), epsilon ($\epsilon$), weight decay ($\lambda$), and learning rate schedule parameters. The performance of an optimizer depends heavily on these choices. Crucially, as Choi et al. (2019b) argue, the *process* used to select these hyperparameters—the tuning protocol—can dramatically influence which optimizer appears superior.

To understand this formally, Choi et al. (2019b, Def 1) introduce the concept of *optimizer inclusion*. An optimizer M is considered a subset or specialization of another optimizer N (denoted $M \subseteq N$) if N can exactly replicate the behavior of M by setting some of its hyperparameters to specific fixed values. For instance, SGD is a subset of Momentum ($SGD \subseteq Momentum$) because Momentum behaves identically to SGD if its momentum coefficient is set to zero. Theoretically, if we had an infinite budget to find the absolute best hyperparameters for both M and N on a given task, the more general optimizer N should never perform worse than its specialization M.

However, practical hyperparameter tuning always operates under limited budgets. We typically cannot afford to run thousands of trials to find the perfect settings. The critical question then becomes: does this theoretical inclusion relationship hold when using realistic, finite tuning budgets (e.g., 20, 50, or 100 trials)? Early empirical studies sometimes suggested it did not, finding, for example, that simpler methods occasionally outperformed more complex ones like Adam (Wilson et al., 2017).

Choi et al. (2019b) investigated this discrepancy directly. They showed that previous results suggesting Adam underperformed Momentum were often due to insufficient tuning protocols. Specifically, if only the learning rate was tuned for both algorithms, Adam might appear worse. However, when additional relevant hyperparameters (like Adam's $\epsilon$ or momentum coefficients) were also tuned using a consistent, fair procedure and budget for both optimizers, the performance differences vanished or even aligned with the theoretical inclusion hierarchy.

> **Figure 3 (from Choi et al., 2019b):**
> ![Tuning Reverses Rankings 1](figures/choi_fig3.png)
> *Caption Quote:* "Figure 3: Tuning more hyperparameters removes the differences in test error between optimizers observed by Wilson et al. (2017). Tuning a subset of optimizer hyperparameters and the initial learning rate is sufficient to equalize performance between all optimizers (left). More extensive hyperparameter tuning in our setup... improves results for all optimizers and still does not produce any differences between optimizer performances (right)." (Choi et al., 2019b, p. 7)

Figure 3 from Choi et al. (2019b) illustrates this finding. The left panel replicates conditions similar to earlier work where limited tuning suggested Adam performed worse than Momentum. The right panel shows that with more comprehensive tuning applied fairly to both optimizers, their performance becomes comparable, contradicting the initial conclusion.

Figure 4 from Choi et al. (2019b) provides further evidence across different workloads and another comparison baseline (Schneider et al., 2019). It shows how the relative ranking of SGD, Momentum, and Adam changes significantly depending on which set of hyperparameters are included in the tuning search space. The rankings only stabilize and become consistent with the inclusion hierarchy ($SGD \subseteq Momentum \subseteq Adam$-variants) when the tuning protocol is sufficiently comprehensive.

> **Figure 4 (from Choi et al., 2019b):**
> ![Tuning Reverses Rankings 2](figures/choi_fig4.png)
> *Caption Quote:* "Figure 4: Tuning more hyperparameters changes optimizer rankings from Schneider et al. (2019) to rankings that are consistent with the inclusion relationships. The leftmost columns for each workload reproduce the rankings from Schneider et al. (2019), while the remaining columns tune over increasingly general search spaces." (Choi et al., 2019b, p. 8)

The conclusion from these results (Choi et al., 2019b, Sec 4.3/5) is that the *tuning protocol is not external to the optimizer; it is an essential part of the algorithm's definition for empirical comparison.* This includes the hyperparameter *search space* considered, the *tuning budget* (number of trials allowed), and the *method* used to sample from the space. Comparing optimizers without explicitly stating and controlling for these aspects of the tuning protocol is fundamentally unreliable and likely to produce results that depend more on the protocol choices than on the optimizers themselves.

## Benchmarking Principles

The challenges outlined—ambiguity in measuring speed, dependence on workload specifics, and sensitivity to hyperparameter tuning protocols—necessitate a standardized approach for comparing optimizers. Without a common framework, drawing reliable conclusions about which algorithms offer genuine improvements is nearly impossible. Rigorous, competitive benchmarks provide a potential solution by establishing clear rules for evaluation.

The **AlgoPerf: Training Algorithms** benchmark (Dahl et al., 2023) represents one such effort to create a framework for systematically comparing neural network training algorithms. Its design incorporates several principles aimed directly at addressing the previously discussed challenges.

First, AlgoPerf uses the **Time-to-Result** metric (Dahl et al., 2023, Sec 4.1). For each task, or "workload," the benchmark defines specific target values for evaluation metrics (like validation error). A submission's performance on that workload is measured as the wall-clock time taken to first reach the target.

Second, to make wall-clock time measurements meaningful, comparisons must occur on **Standardized Hardware** (Dahl et al., 2023, Sec 4.1.2). AlgoPerf specifies a fixed hardware configuration (e.g., a specific type and number of GPUs) on which all official timing runs must be performed. This controls for variations in computational power.

Third, the benchmark includes a **Workload Suite** covering diverse tasks (e.g., image classification, speech recognition, machine translation) to assess the general applicability of an algorithm (Dahl et al., 2023, Sec 4.3). It uses a set of *fixed base workloads*, known to participants, for the primary timing measurements. It also incorporates *randomized held-out workload variants*, sampled after submissions are finalized, to test robustness and discourage overfitting the algorithm design to the specific base workloads.

Fourth, AlgoPerf aims to **Isolate the Training Algorithm** from other parts of the training pipeline (Dahl et al., 2023, Sec 4.2, 4.2.1). It defines a specific Application Programming Interface (API) consisting of functions that a submission must implement (e.g., for updating parameters, initializing optimizer state, selecting data batches). The benchmark system controls the model architecture, data processing, and evaluation logic; submissions can only modify the parts directly related to the training algorithm strategy.

Fifth, addressing the critical issue identified in Section 3, AlgoPerf mandates **Explicit Tuning Rulesets** (Dahl et al., 2023, Sec 4.4). Submissions must compete under one of two predefined procedures for handling hyperparameters:
*   The *External Tuning* ruleset simulates having a limited parallel compute budget for tuning. Submissions provide a hyperparameter search space. The benchmark performs a fixed number of trials (e.g., 5 studies of 5 trials each in the competition described by Kasimbeg et al., 2024) sampling from this space and scores the submission based on the time-to-result achieved by the single best trial found within that budget.
*   The *Self-Tuning* ruleset allows no external tuning. The algorithm must be hyperparameter-free or perform any necessary tuning autonomously *during* the timed training run. These submissions are given a larger runtime budget to accommodate potential internal tuning overhead.

Sixth, since performance is measured across multiple workloads, the benchmark requires a method for **Aggregate Scoring** (Dahl et al., 2023, Sec 4.5). AlgoPerf uses *Performance Profiles*, based on Dolan and Moré (2002), to visualize and summarize performance across the entire workload suite.

A performance profile provides a way to compare multiple submissions simultaneously. The x-axis represents a slowdown factor $\tau$ (where $\tau \ge 1$), indicating how much slower a submission is compared to the *fastest* submission on any given workload. The y-axis shows the fraction (or percentage) of workloads for which the submission successfully reached the target within that slowdown factor $\tau$. Figure 1(b, d) from Kasimbeg et al. (2024) shows an example for the external tuning ruleset. A point $(\tau, y)$ on a submission's line means it solved fraction $y$ of the workloads in at most $\tau$ times the runtime of the best submission for each respective workload. Therefore, optimizers whose lines are higher and further to the left are generally better; they solve more problems faster relative to the best known performance.

> **Figure 1 (from Kasimbeg et al., 2024):**
> ![AlgoPerf Competition Results](figures/kasimbeg_fig1.png)
> *Caption Quote:* "Figure 1: ALGOPERF competition leaderboard & performance profiles for all submissions to the external (top) and self-tuning (bottom) ruleset. The leaderboards (a, c) are ranked by the submissions’ benchmark scores... Higher scores indicate faster training... Note, scores are not comparable between rulesets. In the performance profiles (b, d), each line represents a submission. A step at τ indicates that, for one workload, this submission reaches the target within τ times the runtime of the fastest submission for that workload and ruleset." (Kasimbeg et al., 2024, p. 4)

By incorporating these principles—time-to-result, fixed hardware, diverse workloads, API isolation, explicit tuning rules, and aggregate reporting via performance profiles—benchmarks like AlgoPerf aim to provide a more reliable and informative basis for comparing neural network training algorithms.


## What Was Learned: AlgoPerf Competition Results

The benchmarking principles described in Section 4 were applied in the first AlgoPerf: Training Algorithms competition. The analysis of this competition by Kasimbeg et al. (2024) provides concrete results on the relative performance of various modern optimization strategies under a controlled framework.

Figure 1 summarizes the main outcomes of the competition. Parts (a) and (c) display the final leaderboards for the External Tuning and Self-Tuning rulesets, respectively, ranked by an aggregate benchmark score where higher values indicate better average performance across workloads. Parts (b) and (d) show the corresponding performance profiles, visualizing the trade-off between speed and the fraction of workloads solved. The results clearly differentiate the submitted algorithms, with distinct winners emerging in each ruleset: PYTORCH DISTRIBUTED SHAMPOO led the External Tuning leaderboard, while SCHEDULE FREE ADAMW won the Self-Tuning category. There are noticeable performance gaps between submissions in both rulesets.

One key finding relates to advanced preconditioning methods. The winning submission in the External Tuning ruleset utilized Distributed Shampoo, an algorithm employing non-diagonal preconditioning. Kasimbeg et al. (2024, Sec 3) report that this submission achieved approximately "28% faster model training compared to the baseline" (which used NadamW), averaged across the eight base workloads and measured in wall-clock time. This result suggests that, under the fair tuning conditions imposed by the benchmark, sophisticated preconditioning techniques can offer practical advantages over widely used adaptive methods like AdamW or NadamW.

Another significant result emerged from the Self-Tuning ruleset, which prohibits external hyperparameter tuning. The winning submission, based on Schedule Free AdamW, not only outperformed its baseline but was also competitive with externally tuned methods. Kasimbeg et al. (2024, Sec 3.2) state it was approximately "10% faster than the (external tuning) BASELINE across the seven base workloads both trained successfully." This demonstrates substantial progress in developing algorithms that require less manual tuning, moving closer to the goal of fully automated neural network training.

Table 1 provides a detailed look at the performance of submissions across individual workloads, highlighting the importance of robustness. The table shows normalized runtimes, with gray cells indicating failures (timeout 'inf', errors 'NaN', or disqualification due to held-out workload issues denoted by † or ‡). Many submissions failed to reach the target reliably on all workloads. The winning submissions, while not always the absolute fastest on every task (compare bold entries vs. their rows), demonstrated strong performance *consistently* across the benchmark suite. This suggests that robustness and reliability across diverse tasks, rather than peak speed on a select few, were critical factors for achieving a high overall benchmark score.

> **Table 1 (from Kasimbeg et al., 2024):**
> ![AlgoPerf Per-Workload Runtimes](figures/kasimbeg_table1.png)
> *Caption Quote:* "Table 1: Normalized submission runtimes across all workloads... fastest submission per workload... highlighted in bold. Workload runtimes considered infinite for scoring are marked in gray, with a (suffix) symbol explaining the reason. `inf` denotes that a submission did not reach the workload target... `NaN` indicates an error... A † indicates that a held-out score is ignored due to the submission not reaching the target on the corresponding base workload, while a ‡ indicates that a base workload score is ignored because the submission did not successfully train the associated held-out workload." (Kasimbeg et al., 2024, p. 5)

Finally, the process of running the AlgoPerf competition revealed practical lessons. Kasimbeg et al. (2024, Sec 4) note the substantial engineering effort required to ensure fair comparisons between implementations in different frameworks like JAX and PyTorch, emphasizing the need for functional equivalence and performant implementations. They also discuss methodological trade-offs in benchmark design, such as the computational cost versus the robustness-testing benefits of using held-out workloads (Kasimbeg et al., 2024, Sec 5.1).

## Conclusion

Comparing neural network training algorithms is challenging! As we have seen, simply observing loss curves is insufficient for determining which optimizer is faster due to intersecting trajectories (Section 2). Furthermore, optimizer performance is sensitive not only to the specific workload but also to the hyperparameter tuning protocol employed, including the search space, budget, and sampling method (Section 3). Results obtained without careful control and reporting of these tuning details are difficult to interpret reliably (Choi et al., 2019b).

Standardized benchmarks, such as AlgoPerf (Dahl et al., 2023), are a structured approach to address these difficulties. By fixing workloads, hardware, evaluation metrics (time-to-result), and establishing explicit rules for hyperparameter tuning and API interactions, they provide a more rigorous foundation for empirical comparison (Section 4).

The central message derived from analyzing these challenges and benchmarking efforts is that **an optimizer cannot be evaluated in isolation from its tuning protocol.** For empirical comparisons to be meaningful, the method used to select hyperparameters must be considered an integral part of the algorithm's specification.

Results from the AlgoPerf competition suggest that this benchmark approach is feasible and can yield valuable insights (Kasimbeg et al., 2024). They provide evidence that advanced methods like non-diagonal preconditioning (Distributed Shampoo) and hyperparameter-free algorithms (Schedule Free AdamW) can offer practical advantages over standard methods when evaluated under fair, controlled conditions (Section 5). However, these results also underscore that no single optimizer dominates across all tasks and that robustness is key. Benchmarking is an ongoing effort, with continued refinement needed in areas like workload diversity and cost-efficiency.

For the field to progress effectively, empirical claims about new optimization methods should be supported by evaluations within rigorous benchmark frameworks. When direct benchmarking is not possible, researchers should provide comprehensive details of their experimental setup, including precise hyperparameter tuning protocols, to enable reliable comparisons.

## References

1.  **Choi et al., 2019b:** Dami Choi, Christopher J. Shallue, Zachary Nado, Jaehoon Lee, Chris J. Maddison, and George E. Dahl. (2019). *On Empirical Comparisons of Optimizers for Deep Learning*. arXiv preprint arXiv:1910.05446. [Link](https://arxiv.org/abs/1910.05446)

2.  **Dahl et al., 2023:** George E. Dahl, Frank Schneider, Zachary Nado, Naman Agarwal, Chandramouli Shama Sastry, Philipp Hennig, Sourabh Medapati, Runa Eschenhagen, Priya Kasimbeg, Daniel Suo, Juhan Bae, Justin Gilmer, Abel L. Peirson, Bilal Khan, Rohan Anil, Mike Rabbat, Shankar Krishnan, Daniel Snider, Ehsan Amid, Kongtao Chen, Chris J. Maddison, Rakshith Vasudev, Michal Badura, Ankush Garg, and Peter Mattson. (2023). *Benchmarking Neural Network Training Algorithms*. arXiv preprint arXiv:2306.07179. [Link](https://arxiv.org/abs/2306.07179)

3.  **Dolan & Moré, 2002:** Elizabeth D. Dolan and Jorge J. Moré. (2002). *Benchmarking optimization software with performance profiles*. Mathematical Programming, 91(2), 201-213.

4.  **Kasimbeg et al., 2024:** Priya Kasimbeg, Frank Schneider, Runa Eschenhagen, Juhan Bae, Chandramouli Shama Sastry, Mark Saroufim, Boyuan Feng, Less Wright, Edward Z. Yang, Zachary Nado, Sourabh Medapati, Philipp Hennig, Mike Rabbat, George E. Dahl. (2024). *Accelerating Neural Network Training: An Analysis of the AlgoPerf Competition*. arXiv preprint arXiv:2405.17095. [Link](https://arxiv.org/abs/2405.17095)

5.  **Schneider et al., 2019:** Frank Schneider, Lukas Balles, and Philipp Hennig. (2019). *DeepOBS: A Deep Learning Optimizer Benchmark Suite*. International Conference on Learning Representations (ICLR). [Link](https://arxiv.org/abs/1903.05499)

6.  **Wilson et al., 2017:** Ashia C. Wilson, Rebecca Roelofs, Mitchell Stern, Nathan Srebro, and Benjamin Recht. (2017). *The Marginal Value of Adaptive Gradient Methods in Machine Learning*. Advances in Neural Information Processing Systems (NeurIPS) 30. [Link](https://proceedings.neurips.cc/paper_files/paper/2017/hash/81b3833e2504647f9d794f7d7b9bf341-Abstract.html)

