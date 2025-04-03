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

Having established the structure of large transformer models and the critical role of activation memory, we now turn to the primary strategies used to distribute the training workload across multiple accelerator devices. These techniques allow us to overcome the memory and compute limitations of a single device. This section details the main parallelism paradigms as presented in the Ultrascale Playbook: Data Parallelism, Pipeline Parallelism, Tensor Parallelism, and Fully Sharded Data Parallelism. Each approach offers different trade-offs regarding communication overhead, memory efficiency, and implementation complexity.

### 6.1 Data Parallelism (DP)

Data Parallelism is perhaps the most common and conceptually straightforward approach to distributed training ([Playbook, Section: data_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=data_parallelism)). In DP, each participating worker (typically a GPU) holds a complete, identical copy of the entire model, including all parameters $w$. Training proceeds by splitting a large global data batch $\mathcal{B}$ into smaller micro-batches, $\mathcal{B}_k$, one for each worker $k$.

During each training step, every worker independently performs the forward pass using its local micro-batch $\mathcal{B}\_k$ and the current model parameters $w$ to compute the loss $\sum_{z \in \mathcal{B}\_k} \ell(w, z)$. Subsequently, each worker computes the gradients of this loss with respect to its copy of the model parameters, resulting in local gradients $g\_k = \sum\_{z \in \mathcal{B}\_k} \nabla\_w \ell(w, z)$. To ensure all model replicas remain consistent, these local gradients must be aggregated across all $N$ workers to obtain the gradient for the full global batch, $g = \sum\_{k=1}^N g\_k$. This aggregation is typically achieved using a collective communication operation called AllReduce, which sums the gradient tensors from all workers and distributes the final sum back to every worker. After the AllReduce operation, each worker has the total gradient sum $g$. Each worker then computes the average gradient $\hat{g} = g / \|\mathcal{B}\|$ and performs an identical parameter update (e.g., $w \leftarrow w - \eta \hat{g}$ using an optimizer like SGD or Adam), ensuring all model replicas stay synchronized.

![Figure 1: Data parallelism replicates the model N times across N devices. Each device processes a fraction of the batch and the resulting gradients are averaged across devices using an AllReduce operation.](figures/dp.png)
*Figure 1: Data parallelism replicates the model N times across N devices. Each device processes a fraction of the batch and the resulting gradients are averaged across devices using an AllReduce operation. ([Playbook, Section: data_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=data_parallelism))*

> **Playbook Quote for Figure 1:** "Figure 1: Data parallelism replicates the model N times across N devices. Each device processes a fraction of the batch and the resulting gradients are averaged across devices using an AllReduce operation." ([Playbook, Section: data_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=data_parallelism), figure caption).

The primary communication cost in DP stems from the AllReduce operation performed on the gradients in each step. The time taken by this operation depends on the number of model parameters and the communication bandwidth between the workers.

Data Parallelism is relatively simple to implement and can provide significant training speedups, especially when the communication overhead is small compared to the computation time per step. However, its main limitation is memory usage. Since each worker holds a full replica of the model parameters, the optimizer states, and the activations generated for its micro-batch, DP does not reduce the memory footprint required per device. Therefore, DP alone cannot be used to train models whose parameters and activations exceed the memory capacity of a single worker.