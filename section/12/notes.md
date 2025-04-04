---
layout: course_page
title: Scaling Transformers - Parallelism Strategies from the Ultrascale Playbook
---

# Scaling Transformers: Parallelism Strategies from the Ultrascale Playbook

## Table of contents
1.  [Introduction: The Scaling Challenge](#introduction-the-scaling-challenge)
2.  [Transformers: Anatomy of a Large Model](#transformers-anatomy-of-a-large-model)
3.  [The Memory Bottleneck: Activations in Backpropagation](#the-memory-bottleneck-activations-in-backpropagation)
4.  [Activation Recomputation (Gradient Checkpointing)](#activation-recomputation-gradient-checkpointing)
5.  [Scaling Laws: Why Bigger Models?](#scaling-laws-why-bigger-models)
6.  [Parallelism Strategies](#parallelism-strategies)
    *   [Data Parallelism (DP)](#data-parallelism-dp)
    *   [Pipeline Parallelism (PP)](#pipeline-parallelism-pp)
    *   [Tensor Parallelism (TP)](#tensor-parallelism-tp)
    *   [Fully Sharded Data Parallelism (FSDP / ZeRO)](#fully-sharded-data-parallelism-fsdp--zero)
7.  [Combining Strategies & Conclusion](#combining-strategies--conclusion)

## 1. Introduction: The Scaling Challenge

Modern machine learning often involves training models with billions or even trillions of parameters, particularly within the Transformer architecture family. The computational resources required to train these models, especially regarding memory, frequently exceed the capacity available on a single accelerator device such as a graphics processing unit (gpu). This limitation necessitates the use of distributed training techniques, where the workload is spread across multiple devices working in concert.

Previous lectures examined specific optimization algorithms like sgd and adam (Lectures [6](../6/notes.md), [7](../7/notes.md), [9](../9/notes.md)) and explored the challenges inherent in empirically comparing their performance (Lecture [10](../10/notes.md)). This lecture shifts focus from the optimizer itself to the system-level challenges encountered when training extremely large models. We will explore practical strategies for distributing the training process, primarily drawing guidance from the Hugging Face Ultrascale Playbook ([available here](https://huggingface.co/spaces/nanotron/ultrascale-playbook)), which details methods specifically relevant to scaling transformer models.

This lecture will first provide a minimal overview of the transformer architecture to identify the sources of computational and memory demands. We will then explain the significant memory bottleneck posed by activations during backpropagation. Following this, we will briefly discuss scaling laws as a motivation for training large models. The main part of the lecture will introduce and analyze key parallelism strategies outlined in the playbook: data parallelism (dp), pipeline parallelism (pp), tensor parallelism (tp), and fully sharded data parallelism (fsdp). We will also cover activation recomputation as a critical memory-saving technique.


## 2. Transformers: Anatomy of a Large Model

This section introduces the essential structure of the standard Transformer model. The objective is to identify the components responsible for the model's large parameter count and computational cost, which are critical factors for understanding parallelism strategies. This overview is not exhaustive; for a detailed visual and interactive exploration of the Transformer architecture, the "Transformer Explainer" ([available here](https://poloclub.github.io/transformer-explainer/)) provides an excellent resource. Readers unfamiliar with the basic mechanism should review it.

A Transformer model processes a sequence of input tokens, typically represented as integer indices $x = (x_1, \dots, x_s)$, where $s$ is the sequence length. The overall computation involves an initial embedding layer, followed by a stack of $L$ identical Transformer Blocks, and usually concludes with a task-specific output layer (e.g., projecting to vocabulary logits for language modeling).

The **Embedding Layer** maps the input token indices $x$ to a sequence of continuous vectors $H^{(0)} \in \mathbb{R}^{s \times h}$, where $h$ is the hidden dimension of the model. This is done using a learnable token embedding matrix $W_E \in \mathbb{R}^{V_{size} \times h}$, where $V_{size}$ is the vocabulary size. Each token $x_i$ is mapped to its corresponding row vector in $W_E$. Positional information is then added, typically using fixed or learned positional encodings $P \in \mathbb{R}^{s \times h}$. The resulting initial representation is:

$$
H^{(0)} = \text{Embed}(x) + P
$$

where $\text{Embed}(x)_i = W_E[x_i, :]$ denotes taking the $x_i$-th row of $W_E$. The parameters in this initial stage reside primarily in the embedding matrix $W_E$.

Before describing the Transformer Block, we define **Layer Normalization (LN)**, a technique used throughout the model to stabilize training. Given an input tensor $Z \in \mathbb{R}^{s \times h}$, LN normalizes the activations independently for each token (row) $i$ across the hidden dimension $h$. For the $i$-th token vector $Z_i \in \mathbb{R}^h$, the computation is:

$$
\text{LN}(Z)_i = \gamma \odot \frac{Z_i - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} + \beta
$$

Here, $\mu_i = \frac{1}{h}\sum_{j=1}^h Z_{ij}$ and $\sigma_i^2 = \frac{1}{h}\sum_{j=1}^h (Z_{ij} - \mu_i)^2$ are the mean and variance computed across the hidden dimension $h$ for token $i$. The parameters $\gamma \in \mathbb{R}^h$ and $\beta \in \mathbb{R}^h$ are learnable scale and shift parameters, respectively, and $\epsilon$ is a small constant (e.g., $10^{-5}$) added for numerical stability. The symbol $\odot$ denotes element-wise multiplication. Each LN layer adds $2h$ learnable parameters.

The core of the Transformer consists of $L$ stacked **Transformer Blocks**. Each block $l$ takes an input $H^{(l-1)} \in \mathbb{R}^{s \times h}$ (starting with $H^{(0)}$) and produces an output $H^{(l)} \in \mathbb{R}^{s \times h}$. Within each block, there are two main sub-layers: Multi-Head Self-Attention (MHA) and a Position-wise Feedforward Network (FFN). Residual connections and Layer Normalization are applied around these sub-layers. We describe the common Pre-LN variant, where LN is applied before the input enters each sub-layer.

The **Multi-Head Attention (MHA)** sub-layer allows the model to weigh information from different positions in the sequence.
1.  First, the input is normalized: $X' = \text{LN}_1(H^{(l-1)})$.
2.  This normalized input $X'$ is linearly projected to generate Queries (Q), Keys (K), and Values (Val) using learnable weight matrices $W_Q, W_K, W_V \in \mathbb{R}^{h \times h}$:

    $$
    Q = X' W_Q, \quad K = X' W_K, \quad Val = X' W_V
    $$

3.  To form $a$ attention heads, the resulting $Q, K, Val$ matrices (each in $\mathbb{R}^{s \times h}$) are conceptually split along the hidden dimension into $a$ smaller matrices $Q_i, K_i, Val_i \in \mathbb{R}^{s \times d_k}$, where the head dimension is $d_k = h/a$. This corresponds to partitioning the weight matrices $W_Q, W_K, W_V$.
4.  For each head $i$, scaled dot-product attention is computed:

    $$
    \text{Scores}_i = \text{softmax}_{\text{keys}}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{s \times s}
    $$

    The $\text{softmax}\_{\text{keys}}$ function is applied row-wise to the $s \times s$ matrix of dot products. For a row $z$ of the input matrix, $(\text{softmax}(z))\_j = e^{z\_j} / \sum_{p=1}^{s} e^{z\_p}$, normalizing the attention weights across all key positions for each query position.
5.  The output for head $i$ is a weighted sum of Value vectors:

    $$
    O_i = \text{Scores}_i \cdot Val_i \in \mathbb{R}^{s \times d_k}
    $$

6.  The outputs of all heads are concatenated back together:

    $$
    O_{concat} = \text{Concat}(O_1, \dots, O_a) \in \mathbb{R}^{s \times h}
    $$

7.  Finally, the concatenated outputs are projected using another learnable weight matrix $W_O \in \mathbb{R}^{h \times h}$:

    $$
    O_{MHA} = O_{concat} W_O
    $$


The parameters for the MHA sub-layer are $W_Q, W_K, W_V, W_O$ (and associated biases, often omitted for simplicity or absorbed). The primary computations involve the initial projections, the $QK^T$ matrix multiplication, the $\text{Scores} \cdot Val$ matrix multiplication, and the final projection $W_O$.

After the MHA computation, a **Residual Connection** adds the original input to the MHA output:

$$
H_{intermediate} = H^{(l-1)} + O_{MHA}
$$

The **Position-wise Feedforward Network (FFN)** sub-layer then processes this intermediate result.
1.  First, the input is normalized: $X'' = \text{LN}\_2(H\_{intermediate})$.
2.  The FFN applies two linear transformations with a non-linearity (commonly GELU) in between:

    $$
    O_{FFN} = \text{GELU}(X'' W_1 + b_1) W_2 + b_2
    $$

    Here, $W_1 \in \mathbb{R}^{h \times d_{ff}}$ and $b_1 \in \mathbb{R}^{d_{ff}}$ are parameters for the first linear layer, and $W_2 \in \mathbb{R}^{d_{ff} \times h}$ and $b_2 \in \mathbb{R}^{h}$ are for the second. The intermediate dimension $d_{ff}$ is typically larger than $h$, often $d_{ff}=4h$. The parameters are $W_1, b_1, W_2, b_2$. The computation is dominated by the two large matrix multiplications involving $W_1$ and $W_2$.

A second **Residual Connection** adds the input to the FFN sub-layer to its output, producing the final output of the Transformer block $l$:

$$
H^{(l)} = H_{intermediate} + O_{FFN}
$$

After the final block $L$, the output $H^{(L)}$ is often normalized one last time: 

$$H'_{final} = \text{LN}_{final}(H^{(L)}).$$

An **Output Layer** then typically projects this final representation to the desired output space. For language modeling, this involves a linear layer with weights $W_{LM} \in \mathbb{R}^{h \times V_{size}}$ (often sharing weights with $W_E$):

$$
\text{Logits} = H'_{final} W_{LM} \in \mathbb{R}^{s \times V_{size}}
$$

The entire Transformer model $f$ can be viewed as the composition of these layers: 

$$f(x) = \text{OutputLayer} \circ \text{Block}_L \circ \dots \circ \text{Block}_1 \circ \text{EmbeddingLayer}(x).$$

In summary, the parameters in a standard Transformer model are concentrated in the embedding matrix $W_E$, the MHA projection matrices ($W_Q, W_K, W_V, W_O$ per layer), the FFN weights ($W_1, W_2$ per layer), the LN parameters ($\gamma, \beta$ per normalization), and the final output projection matrix $W_{LM}$. The computational cost is dominated by the large matrix multiplications performed within the MHA mechanism ($QK^T$, $\text{Scores} \cdot Val$) and the FFN layers ($XW_1$, $HW_2$), repeated across $L$ blocks. Understanding this distribution is crucial for designing effective parallelism strategies.

## 3. The Memory Bottleneck: Activations in Backpropagation

PyTorch relies on autodifferentiation (e.g., the formal chain rule) to compute gradients of the loss function $\ell(w, z)$ with respect to the model parameters $w$. Autodifferentiation, or backpropagation as it's called in the deep learning literature, operates by applying the chain rule recursively, starting from the output layer and moving backward through the network. A key requirement of this process is access to intermediate values computed during the forward pass. These intermediate values are often referred to collectively as "activations." Storing these activations consumes significant memory, often becoming the primary bottleneck when training large models. (See Lectures [4](../4/notes.md) and [5](../5/notes.md) for background on computational graphs and automatic differentiation).

Specifically, to compute the gradient with respect to the parameters of a given layer (e.g., the weights $W$ in a linear layer $Y = XW$), backpropagation typically requires the input to that layer during the forward pass (e.g., $X$). Furthermore, to compute the gradient of the loss with respect to the *input* of that layer (needed to propagate gradients further back), one often needs intermediate results computed within that layer (e.g., the output $Y$ or values computed before a non-linearity). Therefore, many intermediate tensors generated during the forward pass must be stored until they are used in the corresponding backward pass computation.

The total memory required to store these necessary activations depends on several factors related to the model architecture and the training configuration. Key factors include:
*   The batch size (`b`): The number of independent sequences processed in parallel.
*   The sequence length (`s`): The number of tokens in each input sequence.
*   The hidden dimension (`h`): The size of the representation vectors within the model.
*   The number of layers (`L`): The depth of the model.
*   Model-specific dimensions like the feedforward intermediate dimension (`d_ff`).
*   The data type used for activations (e.g., float32, float16, bfloat16).

Consider a single transformer block. The input and output tensors $H^{(l-1)}$ and $H^{(l)}$ have dimensions $s \times h$. If processing a batch of size $b$, these tensors become $b \times s \times h$. Many intermediate computations within the MHA and FFN sub-layers (e.g., $Q, K, Val$, attention scores, FFN hidden states) also have dimensions proportional to $b, s, h,$ or $d_{ff}$. Storing these intermediate results across all $L$ layers leads to a total activation memory requirement that scales approximately as:

$$
\text{Activation Memory} \approx C \times L \times b \times s \times h \times (\text{bytes per element})
$$

The constant $C$ depends on the specific transformer architecture and which intermediate values are stored for the backward pass. For standard transformers, $C$ can be significant because values like the attention scores ($b \times a \times s \times s$) or the intermediate FFN activations ($b \times s \times d_{ff}$) must often be kept.

It is crucial to contrast this with the memory required for model parameters. The number of parameters scales primarily with the number of layers $L$ and the square of the hidden dimension $h$ (due to matrices like $W_Q, W_K, W_V, W_O, W_1, W_2$, each roughly $h \times h$ or $h \times d_{ff}$). For large models, parameter count is substantial:


It is crucial to contrast this with the memory required for model parameters. The number of parameters scales primarily with the number of layers $L$ and dimensions related to the model's width, such as the hidden dimension $h$ and the intermediate feedforward dimension $d_{ff}$. Key parameter matrices include the token embeddings ($W_E \in \mathbb{R}^{V_{size} \times h}$), the attention projections ($W_Q, W_K, W_V, W_O \in \mathbb{R}^{h \times h}$ per layer), the FFN weights ($W_1 \in \mathbb{R}^{h \times d_{ff}}, W_2 \in \mathbb{R}^{d_{ff} \times h}$ per layer), and potentially a final output projection ($W_{LM} \in \mathbb{R}^{h \times V_{size}}$). The total parameter memory is approximately:

$$
\begin{aligned}
&\text{Parameter Memory} \\
&\approx (\text{Size}(W_E) + \text{Size}(W_{LM}) + L \times (\text{Size}(W_{Q,K,V,O}) + \text{Size}(W_{1,2}) + \text{Size}(\gamma, \beta)_{\text{LN}})) \\
&\hspace{20pt} \times (\text{bytes per element})
\end{aligned}
$$

Assuming $d_{ff}=4h$ and shared embedding/output weights ($W_{LM} \approx W_E^T$), this scales roughly as $O(V_{size}h + L h^2)$. The critical difference lies in the dependence on batch size `b` and sequence length `s`. Parameter memory is independent of these factors, while activation memory grows linearly with both. In large-scale transformer training, where large batch sizes are used for efficiency and sequences can be long, the activation memory often dwarfs the parameter memory. This makes activation storage the primary constraint limiting the size of the model or the batch size that can fit onto a single accelerator device. Techniques to manage this activation memory are therefore essential for scaling transformer training.

## 4. Activation Recomputation (Gradient Checkpointing)

Activation recomputation, also commonly known as gradient checkpointing, is a technique designed specifically to alleviate the activation memory bottleneck described in Section 3. The core idea is to trade increased computation time for reduced memory usage. Instead of storing all intermediate activations computed during the forward pass that are needed for the backward pass, only a strategically chosen subset of activations is saved. During the backward pass, whenever an activation is required but was not stored, it is recomputed on-the-fly by performing a partial forward pass starting from the nearest previously stored activation. ([Playbook, Section: activation_checkpointing](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=activation_checkpointing)).

For example, within a transformer model, one might choose to store only the inputs to each transformer block ($H^{(l-1)}$) and discard all activations computed *inside* the block (e.g., intermediate MHA and FFN results). When the backward pass needs these internal activations to compute gradients for the block's parameters, the block's forward computation is executed again, starting from the stored $H^{(l-1)}$, to regenerate the needed values just before they are used. This avoids storing the potentially large intermediate tensors from MHA and FFN throughout the entire forward and backward pass of the whole network.

The trade-off is clear: memory usage is significantly reduced because fewer intermediate tensors are stored, but computational cost increases because parts of the forward pass are executed twice (once during the initial forward pass, and again during the backward pass for recomputation). The typical increase in compute time is roughly equivalent to performing one extra forward pass for the segments where recomputation is applied. Activation recomputation is often essential for training very large models or for utilizing larger effective batch sizes than would otherwise fit within the available device memory.

**(Figure Instruction Check):** The playbook section on `activation_checkpointing` contains Figure 12, illustrating the concept.

![Figure 12: Activation Checkpointing. Checkpointing trades compute for memory by recomputing intermediate activations during the backward pass.](figures/activation_checkpointing.png)
*Figure 12: Activation Checkpointing. Checkpointing trades compute for memory by recomputing intermediate activations during the backward pass. ([Playbook, Section: activation_checkpointing](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=activation_checkpointing))*

> **Playbook Quote for Figure 12:** "Figure 12: Activation Checkpointing. Checkpointing trades compute for memory by recomputing intermediate activations during the backward pass. The top shows a standard forward and backward pass where all activations are stored. The bottom shows activation checkpointing where only the input to the checkpointed segment is stored, and activations within the segment are recomputed during the backward pass." ([Playbook, Section: activation_checkpointing](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=activation_checkpointing), adapted from figure caption and surrounding text).

## 5. Primer: Distributed Communication Primitives

Distributing model training across multiple devices, such as gpus, necessitates communication between these independent processing units, often referred to as workers. Standard patterns for this inter-worker communication and synchronization are known as collective operations. Understanding these basic operations is essential for grasping how different parallelism strategies function. This section provides a brief overview of common communication primitives, drawing from the concepts outlined in the Ultrascale Playbook appendix ([Playbook, Appendix A0: Parallel Programming Crash Course](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=appendix_a0_parallel_programming_crash_course)).

We consider a group of $N$ workers involved in the distributed computation. Each worker is assigned a unique integer identifier, its rank, typically ranging from 0 to $N-1$. Some collective operations involve a designated root worker, or specify source (`src`) and destination (`dst`) ranks for the data transfer.

The **Broadcast** operation involves one designated source worker (`src`) sending an identical copy of its data tensor to all other workers in the group, including itself. This is commonly used to distribute initial model parameters or configuration settings from one worker (e.g., rank 0) to all others at the beginning of training.

*(Instruction: Insert Figure 5 placeholder here)*
**Figure 5: Broadcast Operation**
> **Playbook Quote for Broadcast Figure:** "The broadcast operation does just that: [shows data moving from Node 1 to all others]" ([Playbook, Appendix A0: Parallel Programming Crash Course](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=appendix_a0_parallel_programming_crash_course), adapted from figure description).

The **Reduce** operation aggregates data from all workers. Each worker provides an input tensor, and a specified reduction function (e.g., SUM, AVG, MAX, MIN) is applied element-wise across these input tensors. The final tensor containing the reduced result is stored only on a single, designated destination worker (`dst`). This allows, for example, summing partial results computed across workers onto one main worker.

The **AllReduce** operation also performs an element-wise reduction across input tensors from all workers, using a specified function like SUM. However, unlike Reduce, the final reduced tensor result is then distributed back to *all* workers in the group. This primitive is fundamental to Data Parallelism (discussed in Section 6.1), where it is used to average the gradients computed independently on each worker.

*(Instruction: Insert Figure 6 placeholder here)*
**Figure 6: Reduce and AllReduce Operations**
> **Playbook Quote for Reduce/AllReduce Figure:** "In the Reduce paradigm the result is sent to the root node only, whereas in the AllReduce case the result is broadcasted to all nodes" ([Playbook, Appendix A0: Parallel Programming Crash Course](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=appendix_a0_parallel_programming_crash_course)).

The **Gather** operation collects distinct data chunks from different workers onto a single worker. Each worker $k$ initially holds its own tensor $T_k$. In a Gather operation, all these tensors $T_0, T_1, ..., T_{N-1}$ are sent to a designated destination worker (`dst`), which then holds the collection of all tensors (e.g., concatenated or as a list).

The **AllGather** operation performs the same collection process as Gather, gathering the distinct tensor $T_k$ from each worker $k$. However, the resulting collection of all tensors $(T_0, ..., T_{N-1})$ is made available to *all* workers in the group, not just the destination worker. This is useful when all workers need access to the complete set of data distributed across the group, such as gathering sharded parameters in certain parallelism schemes.

*(Instruction: Insert Figure 7 placeholder here)*
**Figure 7: Gather and AllGather Operations**
> **Playbook Quote for Gather/AllGather Figure:** "...gather all data on one node (in case of Gather) or gather all data on all nodes (in the case of AllGather)." ([Playbook, Appendix A0: Parallel Programming Crash Course](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=appendix_a0_parallel_programming_crash_course)).

The **Scatter** operation is functionally the inverse of Gather. A single designated source worker (`src`) holds a collection of distinct data chunks (e.g., a list of tensors or a partitioned tensor). It sends exactly one distinct chunk to each worker in the group, including itself. This distributes different pieces of data from one source to multiple destinations.

The **ReduceScatter** operation combines aggregation and scattering. Each worker $k$ starts with a collection of data chunks, where chunk $j$ is notionally intended for worker $j$. First, for each chunk index $j$, the corresponding chunks from *all* workers are reduced together (e.g., summed element-wise). Then, the resulting reduced chunk $j$ is sent *only* to worker $j$. This is used, for instance, in Fully Sharded Data Parallelism (Section 6.4) to compute and distribute shards of the averaged gradients efficiently.

*(Instruction: Insert Figure 8 placeholder here)*
**Figure 8: Scatter and ReduceScatter Operations**
> **Playbook Quote for Scatter/ReduceScatter Figure:** "Scatter operation is to take data on one node and distribute slices of it... ReduceScatter pattern is slightly more complex: imagine you apply an operation like in the Reduce case but instead of moving the result to just one node we also distribute it evenly to all nodes" ([Playbook, Appendix A0: Parallel Programming Crash Course](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=appendix_a0_parallel_programming_crash_course), adapted).

The **Barrier** operation provides explicit synchronization. When a worker calls barrier, it pauses execution until all other workers in the group have also called barrier. Once all workers have reached the barrier, they are all allowed to proceed. This ensures that certain stages of computation are complete across all workers before the next stage begins, although excessive use can introduce unnecessary delays by forcing faster workers to wait for slower ones.

*(Instruction: Insert Figure 9 placeholder here)*
**Figure 9: Barrier Operation**
> **Playbook Quote for Barrier Figure:** "A barrier is not lifted until all nodes have reached it. Then only are they allowed to continue..." ([Playbook, Appendix A0: Parallel Programming Crash Course](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=appendix_a0_parallel_programming_crash_course)).

These collective operations are typically implemented in specialized communication libraries. For distributed GPU training, the NVIDIA Collective Communications Library (NCCL) is widely used. NCCL provides highly optimized implementations of these primitives for efficient communication directly between GPUs, often leveraging high-speed interconnects like NVLink. Frameworks like PyTorch utilize NCCL as a backend to perform these distributed communication tasks during training.

## 6. Parallelism Strategies

Having introduced the fundamental distributed communication primitives in Section 5, we now explore the primary parallelism strategies employed to train large models across multiple devices. These strategies distribute the computational workload and model state in distinct ways, each presenting specific advantages and limitations concerning memory usage, compute efficiency, and communication overhead. The techniques described here follow the exposition in the Ultrascale Playbook ([Playbook Link Here](https://huggingface.co/spaces/nanotron/ultrascale-playbook)).

### 6.1 Data Parallelism (DP) & Optimizations

Data Parallelism (DP) is a common strategy where the complete model is replicated on each participating worker device (e.g., gpu). The global data batch $\mathcal{B}$ is divided into smaller micro-batches, $\mathcal{B}_k$. Each worker $k$ independently processes its assigned micro-batch $\mathcal{B}_k$ through the forward and backward passes using its local model replica to compute local gradients $\nabla \ell(w, \mathcal{B}_k)$. ([Playbook, Section: data_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=data_parallelism)).

Since gradients are computed on different data subsets, they must be synchronized across all workers before the parameters $w$ are updated. This ensures that all model replicas remain consistent. Typically, the gradients are averaged across all workers using an AllReduce collective operation (Section 5). The synchronized gradient $\nabla L(w, \mathcal{B}) = \frac{1}{N_d} \sum_{k=1}^{N_d} \nabla \ell(w, \mathcal{B}_k)$ (where $N_d$ is the number of DP workers) is then used by each worker to perform an identical optimizer step.

*(Instruction: Insert Figure 10 placeholder here)*
**Figure 10: Data Parallelism Overview**
> **Playbook Quote for Figure 10:** "The idea behind data parallelism (DP) is to replicate the model on several GPUs... gradients from the model instances will be averaged using an operation called ‘all-reduce’" ([Playbook, Section: data_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=data_parallelism), adapted from figure and text).

A naive DP implementation would perform the AllReduce only after the entire backward pass completes on all workers. This leaves gpus idle during the communication phase. Performance is improved by overlapping the AllReduce communication with the backward pass computation. As soon as the gradients for a subset of parameters are computed (e.g., for the final layers of the model), the AllReduce operation for those specific gradients can be initiated while the backward pass continues computing gradients for earlier layers. This overlap is often achieved using framework-specific hooks attached to parameter gradients.

*(Instruction: Insert Figure 11 placeholder here)*
**Figure 11: Data Parallelism with Computation-Communication Overlap**
> **Playbook Quote for Figure 11:** "...gradients (red boxes) for a layer can be gathered and summed even before the gradients from earlier layers... have been computed... Overlapping computation and communication reduces the time spent waiting for gradient synchronization..." ([Playbook, Section: data_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=data_parallelism), adapted from figure and text).

Further efficiency gains can be achieved through gradient bucketing. Instead of initiating a separate AllReduce operation for each parameter or small group of parameters as soon as their gradients are ready, gradients are collected into larger buffers, or "buckets." A single AllReduce operation is then performed for each bucket. This reduces the number of distinct communication calls, which can lower overhead, especially on networks where latency is a factor.

*(Instruction: Insert Figure 12 placeholder here)*
**Figure 12: Data Parallelism with Gradient Bucketing**
> **Playbook Quote for Figure 12:** "...group gradients into buckets and launch a single all-reduce for all the gradients within the same bucket instead of performing independent all-reduce for each gradient." ([Playbook, Section: data_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=data_parallelism), adapted from figure and text).

When using gradient accumulation (performing multiple forward/backward passes for several micro-batches before one optimizer step), the AllReduce synchronization should only occur after the gradients for *all* accumulated micro-batches have been computed and summed locally. Synchronization after each intermediate micro-batch's backward pass is unnecessary and adds overhead. Frameworks typically provide mechanisms, like a `no_sync()` context manager, to disable gradient synchronization during the intermediate steps of gradient accumulation.

Data parallelism is conceptually straightforward and effectively parallelizes computation over the data dimension, leading to increased training throughput. However, its primary limitation is memory usage: every worker must store the entire model, its gradients, the optimizer states, and the activations for its micro-batch. Consequently, DP alone cannot be used if the model itself is too large to fit on a single worker. Additionally, the cost of the AllReduce operation scales with model size and can become a bottleneck as the number of workers ($N_d$) grows large.

### 6.2 ZeRO (Zero Redundancy Optimizer) Stages

The ZeRO (Zero Redundancy Optimizer) techniques aim to overcome the memory limitations of standard Data Parallelism by eliminating the redundant storage of model state (optimizer states, gradients, and parameters) across DP workers. Instead of replication, ZeRO partitions these states, assigning each DP worker responsibility for only a fraction ($1/N_d$) of the total state. ([Playbook, Section: zero](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=zero)).

The core idea is partitioning. While activations necessarily differ across DP workers (due to different input data) and cannot be sharded this way, the other three components are identical across replicas in standard DP after synchronization and are thus candidates for partitioning. ZeRO implements this partitioning in three progressive stages.

ZeRO Stage 1 partitions only the optimizer states. Each worker holds $1/N_d$ of the states (e.g., Adam momentum/variance). During training, the backward pass computes full gradients locally. These gradients are then reduced using ReduceScatter (Section 5), such that worker $k$ receives the final, summed gradient shard corresponding only to the parameters whose optimizer states it manages. Worker $k$ performs the optimizer update only on its local parameter shard (often maintained in FP32 for precision). Finally, an AllGather collective (Section 5) is required to reassemble the complete, updated set of parameters (typically in the working precision like BF16) on all workers before the next forward pass begins.

*(Instruction: Insert Figure 13 placeholder here)*
**Figure 13: ZeRO-1 Communication Pattern**
> **Playbook Quote for Figure 13:** [Find quote in playbook describing the ZeRO-1 figure, highlighting ReduceScatter for grads and AllGather for params after optimizer step.] ([Playbook, Section: zero](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=zero)).

ZeRO Stage 2 partitions both the optimizer states and the gradients. The key change from ZeRO-1 is that the backward pass communication becomes a ReduceScatter directly. This means each worker only materializes the $1/N_d$ portion of the summed gradients it needs for its optimizer state shard, saving memory compared to ZeRO-1 which required temporary storage of full gradients before the ReduceScatter. The subsequent optimizer step and parameter AllGather remain similar to ZeRO-1.

ZeRO Stage 3, often referred to as Fully Sharded Data Parallelism (FSDP) in PyTorch implementations, partitions all three components: parameters, gradients, and optimizer states. Each worker persistently stores only its assigned $1/N_d$ shard of the parameters. During the forward pass, as computation proceeds layer by layer, each worker uses an AllGather operation to temporarily retrieve the necessary full parameters for the current layer, performs the computation, and then immediately discards the parameter shards it does not own. The backward pass operates similarly, gathering parameters as needed and computing gradients only for the locally owned parameter shard. Gradient synchronization occurs via ReduceScatter, as in ZeRO-2.

*(Instruction: Insert Figure 14 placeholder here)*
**Figure 14: ZeRO-3 / FSDP Parameter Handling (Forward and Backward)**
> **Playbook Quote for Figure 14:** "So as we perform the forward pass... we retrieve the necessary parameters on demand and immediately flush them... The backward pass works the same way..." ([Playbook, Section: zero](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=zero), adapted from `dp_zero3_fwd.svg` / `dp_zero3_bwd.svg` descriptions).

The progressive partitioning offered by ZeRO stages significantly reduces the memory required per worker for model state, allowing DP techniques to be applied to much larger models. ZeRO-3 offers the maximum savings for model state components.

*(Instruction: Insert Figure 15 placeholder here)*
**Figure 15: ZeRO Memory Savings Overview**
> **Playbook Quote for Figure 15:** [Find quote in playbook explaining the `zero_memory.svg` diagram, showing memory reduction formulas or trends for parameters ($\Psi$), optimizer states (k$\Psi$), and gradients across ZeRO stages and DP degree $N_d$.] ([Playbook, Section: zero](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=zero)).

The primary advantage of ZeRO is the substantial reduction in memory required per worker for parameters, gradients, and optimizer states. This enables training larger models using data parallelism. However, ZeRO introduces more communication compared to standard DP. ZeRO-1 and ZeRO-2 change the gradient AllReduce to a ReduceScatter and add a parameter AllGather. ZeRO-3 replaces the single parameter AllGather with potentially many AllGather operations throughout the forward and backward passes (one per layer or block being gathered). The performance of ZeRO-3 relies heavily on the ability to effectively overlap this parameter gathering (prefetching) with computation. Importantly, none of the ZeRO stages partition the activation memory.

### 6.3 Tensor Parallelism (TP)

Tensor Parallelism (TP) addresses scenarios where even a single layer's weights or activations exceed the memory of one device, or where further parallelization *within* a layer is desired to speed up computation. TP achieves this by partitioning the execution of individual large operations, such as the matrix multiplications found in transformer MHA and FFN layers, across multiple workers. These workers compute on shards of the tensors involved in the operation. ([Playbook, Section: tensor_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism)).

Consider a linear layer computation $Y = XW$. TP typically splits the weight matrix $W$ and performs corresponding computations on shards:
*   **Column Parallelism:** $W$ is split column-wise across $N\_{tp}$ workers: $W = [W\_1 \| W\_2 \| \dots \| W\_{N\_{tp}}]$. Each worker $k$ computes $Y\_k = X W\_k$ using the *full* input $X$ (which must be available to all TP workers). The final output $Y$ exists logically as the concatenation $[Y\_1 \| \dots \| Y\_{N_{tp}}]$ across the workers' memories.
*   **Row Parallelism:** $W$ is split row-wise: $W = [W\_1^T \| \dots \| W_{N\_{tp}}^T]^T$. The input $X$ is also considered split row-wise (often matching the output sharding of a preceding column-parallel layer). Each worker $k$ computes a partial result $Y_k = X\_k W\_k$. The final output $Y = \sum\_{k=1}^{N\_{tp}} Y_k$ is obtained by summing the partial results using an AllReduce collective across the TP workers.

*(Instruction: Insert Figure 16 placeholder here)*
**Figure 16: Tensor Parallelism for Linear Layers (Column and Row)**
> **Playbook Quote for Figure 16:** [Find quote in playbook describing the TP diagram(s) for linear layers, explaining column splitting requiring broadcast/identity on input and row splitting requiring AllReduce on output.] ([Playbook, Section: tensor_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism)).

In transformer blocks, FFN layers commonly use a Column-Parallel followed by a Row-Parallel layer. This configuration avoids an AllReduce between the two FFN layers. MHA layers apply similar partitioning: column parallelism for the $W_Q, W_K, W_V$ projections (effectively giving each worker a subset of attention heads) and row parallelism for the output projection $W_O$. A practical constraint is that the TP degree ($N_{tp}$) should ideally divide the number of attention heads evenly.

*(Instruction: Insert Figure 17 placeholder here)*
**Figure 17: Tensor Parallelism Applied to Transformer FFN and MHA**
> **Playbook Quote for Figure 17:** [Find quote in playbook describing the application of column/row parallelism within FFN and MHA blocks.] ([Playbook, Section: tensor_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism)).

TP necessitates communication collectives (like AllReduce for row parallelism, potentially identity/broadcast/AllGather depending on exact data flow for column parallelism) *within* the forward and backward computation of a single layer. This demands high-bandwidth, low-latency interconnects, making TP most effective within a single compute node equipped with connections like NVLink.

*(Instruction: Insert Figure 18 placeholder here)*
**Figure 18: Communication Timeline in Tensor Parallelism**
> **Playbook Quote for Figure 18:** "...we hit a synchronization point with the AllReduce operation that cannot be overlapped with computation. This exposed communication overhead is necessary to combine partial results across tensor-parallel ranks..." ([Playbook, Section: tensor_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism)).

Tensor Parallelism reduces memory requirements for weights, gradients, optimizer states, *and* activations within the parallelized layers, as these are sharded across the TP workers. It can also reduce the wall-clock time per layer if the computation savings outweigh the communication overhead. However, this communication overhead is significant and limits TP's scalability, especially across nodes with slower interconnects. TP also increases implementation complexity and does not naturally parallelize all layer types (like Layer Normalization), which still require access to the full hidden dimension representation, limiting overall activation memory savings.

### 6.4 Sequence Parallelism (SP)

Sequence Parallelism (SP) is an extension designed to work alongside Tensor Parallelism (TP). It addresses a limitation of TP: operations like Layer Normalization or Dropout typically require the input tensor to be complete along the hidden dimension ($h$), preventing TP's hidden-dimension sharding from reducing activation memory for these specific operations. SP mitigates this by sharding the activations along the *sequence* dimension ($s$) for these particular operations, while TP handles the hidden dimension sharding for matrix multiplications. ([Playbook, Section: sequence_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=sequence_parallelism)).

SP involves partitioning the input sequence across the workers participating in the TP group. Operations like LayerNorm can then be applied locally to each sequence chunk. However, transitions are required when moving between TP-sharded regions (where tensors are split along dimension $h$) and SP-sharded regions (split along dimension $s$). To enter a TP region (like a column-parallel linear layer), an AllGather operation is performed along the sequence dimension to reconstruct the full tensor needed by the TP layer. To exit a TP region (like after a row-parallel linear layer whose output needs to be sharded by sequence for the subsequent LayerNorm), a ReduceScatter operation is used along the sequence dimension. This ReduceScatter simultaneously performs the necessary reduction (summing partial results from the row-parallel layer) and distributes the result correctly sharded by sequence dimension.

*(Instruction: Insert Figure 19 placeholder here)*
**Figure 19: Tensor Sharding and Communication in TP vs. TP+SP**
> **Playbook Quote for Figure 19:** [Find quote in playbook explaining the figure with f/f*/g/g*, describing the transitions: "g" operation (all-gather) combines... back to full sequence length... "g*" operation (reduce-scatter) which reduces... while scattering along sequence dimension".] ([Playbook, Section: sequence_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=sequence_parallelism)).

The communication volume with SP is comparable to TP alone, but the pattern shifts to using AllGather and ReduceScatter at the boundaries between TP and SP execution regions. It still demands high-bandwidth intra-node interconnects.

The main advantage of Sequence Parallelism is that it further reduces the peak activation memory footprint compared to using only Tensor Parallelism, by allowing activations for operations like LayerNorm to remain sharded along the sequence dimension. This can enable larger batch sizes or sequence lengths within a TP group. However, SP inherits the communication bottlenecks and intra-node scaling limitations of TP and adds another layer of implementation complexity due to the sharding transitions.

### 6.5 Context Parallelism (CP) & Ring Attention

Context Parallelism (CP) is designed to tackle the activation memory challenge posed by extremely long input sequences. Even with TP+SP and activation recomputation, the memory required can become prohibitive for sequence lengths in the tens or hundreds of thousands, as certain parts of the computation within TP still effectively process the full sequence length. CP addresses this by partitioning the input sequence across workers *globally* for nearly all layers, not just specific ones like in SP. ([Playbook, Section: context_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=context_parallelism)).

Most operations in a transformer (like FFNs, LayerNorm) can operate independently on their local sequence chunk when the input is sharded along the sequence dimension. The main challenge lies in the self-attention mechanism (MHA), where each token (query) potentially needs to attend to all other tokens (keys and values) in the sequence. If the sequence is distributed, workers must communicate to exchange the necessary Key (K) and Value (Val) information.

Ring Attention provides an efficient communication pattern for MHA under Context Parallelism. Workers are arranged in a logical ring. Each worker computes attention scores using its local Query (Q) chunk against the Key (K) and Value (Val) chunks it currently holds. Simultaneously, it sends its current K/V chunk to the next worker in the ring and receives the K/V chunk from the previous worker asynchronously. This allows the communication of K/V chunks to be overlapped with the local attention computation, minimizing idle time.

*(Instruction: Insert Figure 20 placeholder here)*
**Figure 20: Ring Attention Mechanism**
> **Playbook Quote for Figure 20:** [Find quote in playbook describing the ring attention animation/diagram, explaining the passing of K/V chunks and overlap: "each GPU first initiates an asynchronous communication... computes the attention score... receives keys and values from the previous GPU..."] ([Playbook, Section: context_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=context_parallelism)).

For causal attention masks used in decoder models, simply splitting the sequence sequentially can lead to computational load imbalance (early sequence chunks require less computation). ZigZag attention partitioning distributes chunks non-sequentially across workers to achieve better load balance during the Ring Attention computation.

*(Instruction: Insert Figure 21 placeholder here)*
**Figure 21: ZigZag Partitioning for Load Balancing in Ring Attention**
> **Playbook Quote for Figure 21:** "...assigning the tokens not purely sequential to the GPUs but by mixing the ordering... computation is now balanced across all GPUs." ([Playbook, Section: context_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=context_parallelism), adapted from `cp_zigzagmask.svg` description).

Communication in CP primarily involves the ring-based exchange of K/V chunks within the attention layers. Additionally, similar to Data Parallelism (since each worker processes different parts of the sequence), an AllReduce operation is needed to synchronize gradients across the CP group before the optimizer step.

Context Parallelism enables training on exceptionally long sequences by sharding activation memory along the sequence dimension throughout the model. Ring Attention offers an efficient mechanism to handle the necessary communication within the attention layers. However, it adds complexity to the attention mechanism and still requires gradient synchronization across the group.

### 6.6 Pipeline Parallelism (PP) & Scheduling

Pipeline Parallelism (PP) is primarily motivated by the need to train models whose parameters are too large to fit even on a group of tensor-parallel workers within a single node, or to scale training across a large number of nodes where TP's communication overhead becomes prohibitive. PP partitions the model's layers *sequentially* into stages. Each stage, consisting of a contiguous block of layers, is assigned to a different worker or group of workers. Data flows through the pipeline, with the output of stage $i$ becoming the input for stage $i+1$. ([Playbook, Section: pipeline_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=pipeline_parallelism)).

A naive implementation where a full batch passes completely through stage 1, then stage 2, and so on, is highly inefficient. Only one stage is computationally active at any given time, leaving all other stages idle. This idle time is known as the pipeline bubble. The size of the bubble relative to the useful computation time increases linearly with the number of pipeline stages ($p$), drastically reducing hardware utilization.

*(Instruction: Insert Figure 22 placeholder here)*
**Figure 22: Pipeline Bubble in Naive Sequential Execution**
> **Playbook Quote for Figure 22:** "An example of Pipeline parallelism... The remaining idle time is indicated in grey and usually called the 'bubble'..." ([Playbook, Section: pipeline_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=pipeline_parallelism)).

To mitigate the bubble, the global data batch is split into multiple smaller micro-batches ($m$). These micro-batches are processed through the pipeline stages in a staggered manner, allowing multiple stages to operate concurrently on different micro-batches. Different schedules exist for managing the flow of these micro-batches:
*   **All-Forward-All-Backward (AFAB):** All micro-batches complete their forward pass through all stages before any backward pass begins. This is relatively simple to implement but requires storing activations for all $m$ micro-batches until the backward phase begins, potentially leading to high peak activation memory. The relative bubble size is reduced to approximately $(p-1)/m$.
*   **One-Forward-One-Backward (1F1B):** This schedule interleaves forward and backward passes more tightly. As soon as a micro-batch finishes its forward pass through the last stage, its backward pass begins, propagating backward through the stages. This allows subsequent micro-batches' forward passes to overlap with preceding micro-batches' backward passes. 1F1B significantly reduces peak activation memory compared to AFAB (roughly proportional to $p$ micro-batches instead of $m$) but maintains a similar bubble size of $(p-1)/m$ and increases scheduling complexity.

*(Instruction: Insert Figure 23 placeholder here)*
**Figure 23: Pipeline Parallelism Schedules (AFAB and 1F1B)**
> **Playbook Quote for Figure 23:** [Find quotes describing the AFAB (`pp_afab2.svg`) and 1F1B (`image.png` with 1F1B schedule) figures, highlighting micro-batch flow and activation storage needs.] ([Playbook, Section: pipeline_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=pipeline_parallelism)).

More advanced schedules exist. **Interleaved Stages** assign non-contiguous layers (e.g., odd layers to stage 1, even layers to stage 2) to potentially reduce the bubble further, but increase communication between stages. Techniques like **ZeroBubble** or **DualPipe** aim to nearly eliminate the bubble by decomposing the backward pass computation (separating input-gradient calculation from weight-gradient calculation) and creating highly optimized, fine-grained schedules, albeit with significant implementation complexity.

Communication in PP primarily involves point-to-point transfers between adjacent pipeline stages: sending activations forward and gradients backward for each micro-batch. This communication pattern is typically less demanding on network bandwidth, especially across nodes, compared to the collective operations used frequently in TP or FSDP.

Pipeline Parallelism enables scaling model depth across many devices and is less sensitive to inter-node communication bandwidth than TP. However, it introduces the pipeline bubble inefficiency, which limits ideal scaling. Effective implementation requires complex scheduling and careful load balancing of computation across stages. Activation memory usage depends critically on the chosen schedule.

### 6.7 Expert Parallelism (EP)

Expert Parallelism (EP) is a specialized technique applicable only to Mixture-of-Experts (MoE) model architectures. MoE models replace some standard FFN layers with a larger set of parallel "expert" FFNs. For each input token, a routing mechanism (a "gate") selects a small subset of these experts (often just one or two) to process that token. ([Playbook, Section: expert_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=expert_parallelism)).

*(Instruction: Insert Figure 24 placeholder here)*
**Figure 24: Mixture-of-Experts (MoE) Layer Concept**
> **Playbook Quote for Figure 24:** "Illustrationg [sic] of a MoE layer taken from the Switch Transformers paper" ([Playbook, Section: expert_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=expert_parallelism)).

Expert Parallelism leverages this architecture by assigning different expert networks to different workers. Since the experts operate independently, they can be distributed. During the forward pass, after the router determines which expert(s) each token should be sent to, an AlltoAll collective communication operation is used. This operation efficiently routes token representations from their current worker to the worker(s) hosting their assigned expert(s). After computation by the expert(s), another AlltoAll operation gathers the results back to the original workers.

Expert Parallelism alone only parallelizes the MoE layers. Other layers (like attention) would perform redundant computations on all workers if only EP were used. Therefore, EP is almost always combined with Data Parallelism. In such a setup, the workers are typically arranged in a 2D grid. One dimension handles Data Parallelism (replicating the non-MoE parts and processing different data micro-batches), while the other dimension handles Expert Parallelism (distributing the experts within each DP replica).

*(Instruction: Insert Figure 25 placeholder here)*
**Figure 25: Combining Expert Parallelism (EP) and Data Parallelism (DP)**
> **Playbook Quote for Figure 25:** [Find quote in playbook describing the `ep_schema.png` figure, illustrating the combination of DP and EP.] ([Playbook, Section: expert_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=expert_parallelism)).

Expert Parallelism allows models to scale to extremely large parameter counts (by having many experts) while keeping the computational cost per token relatively low (since only a few experts are active). The main drawback is the communication overhead introduced by the AlltoAll operations needed for routing tokens, which can become significant depending on the network and routing patterns. It is only applicable to MoE model architectures.