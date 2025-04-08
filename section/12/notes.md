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

This lecture will first provide a minimal overview of the transformer architecture to identify the sources of computational and memory demands. We will then explain the significant memory bottleneck posed by activations during backpropagation. Following this, we give a primer on distributed computing primitives. The main part of the lecture will introduce and analyze key parallelism strategies outlined in the playbook: data parallelism (dp), pipeline parallelism (pp), tensor parallelism (tp), and fully sharded data parallelism (fsdp). We will also cover activation recomputation as a critical memory-saving technique.


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

Having established the structure of large transformer models, the activation memory bottleneck, and the utility of activation recomputation, we now turn to the primary strategies used to distribute the training workload across multiple accelerator devices. These techniques, detailed in the Hugging Face Ultrascale Playbook ([Playbook Link Here](https://huggingface.co/spaces/nanotron/ultrascale-playbook)), allow us to overcome the memory and compute limitations of a single device by partitioning the model state and computation in different ways.

### 6.1 Data Parallelism (DP)

Data Parallelism (DP) is a widely used strategy where the complete set of model parameters, denoted by $w$, is replicated on each of $N\_d$ participating worker devices (e.g., gpus) in a DP group. Training involves processing a large global data batch $\mathcal{B}$. This batch is divided into $N_d$ smaller micro-batches, $\mathcal{B}\_k$, where $k \in \{1, \dots, N_d\}$. Each worker $k$ independently performs the forward pass using its local micro-batch $\mathcal{B}\_k$ and its local replica of the parameters $w$ to compute the loss for its data portion. Subsequently, it performs the backward pass to compute the local gradient $g_k = \frac{1}{\|\mathcal{B}\_k\|} \sum\_{z \in \mathcal{B}_k} \nabla_w \ell(w, z)$. ([Playbook, Section: data_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=data_parallelism)).

Because each worker computes gradients based on a different subset of the data ($\mathcal{B}\_k$), these local gradients $g_k$ must be synchronized across all workers before updating the model parameters $w$. This synchronization ensures that all model replicas remain identical. The goal is to compute the average gradient over the entire global batch $\mathcal{B}$:

$$
\hat{g} = \frac{1}{|\mathcal{B}|} \sum_{k=1}^{N_d} \sum_{z \in \mathcal{B}_k} \nabla_w \ell(w, z) = \frac{1}{N_d} \sum_{k=1}^{N_d} g_k
$$

This is typically achieved using an `AllReduce` collective communication operation (Section 5) that sums the local gradients $g_k$ from all workers and distributes the total sum $\sum_k g_k$ back to every worker. Each worker then divides this sum by $N_d$ (or equivalently, by $\|\mathcal{B}\|$) to obtain $\hat{g}$. Finally, each worker applies an identical optimizer step using this averaged gradient, for instance:

$$
w_{t+1} = \text{OptimizerStep}(w_t, \hat{g})
$$

*(Instruction: Place Figure 10 from Playbook PDF here)*
![Figure 10: Data parallelism replicates the model N times across N devices. Each device processes a fraction of the batch and the resulting gradients are averaged across devices using an AllReduce operation.](figures/dp.png)
*Figure 10: Data parallelism replicates the model N times across N devices. Each device processes a fraction of the batch and the resulting gradients are averaged across devices using an AllReduce operation. ([Playbook, Section: data_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=data_parallelism))*

> **Playbook Quote for Figure 10:** "Figure 10: Data parallelism replicates the model N times across N devices. Each device processes a fraction of the batch and the resulting gradients are averaged across devices using an AllReduce operation." ([Playbook, Section: data_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=data_parallelism), adapted from figure caption and text).

A simple implementation might perform the `AllReduce` operation only after the entire backward pass computation for $g\_k$ is complete on all workers. This approach, however, leaves the communication network idle during computation and the compute units idle during communication. Performance can often be improved by overlapping communication and computation. As the backward pass proceeds from the final layers towards the initial layers, gradients for parameters in later layers become available earlier. The `AllReduce` operation for these gradients can be initiated while the backward pass continues computing gradients for parameters in earlier layers. This requires mechanisms, often provided by distributed training frameworks, to track gradient readiness and initiate communication asynchronously.

*(Instruction: Place Figure 11 from Playbook PDF here)*
![Figure 11: Data Parallelism with Communication Overlap](figures/dp_overlap.png)
*Figure 11: Data Parallelism with Computation-Communication Overlap. Gradients (red boxes) for later layers can be communicated while gradients for earlier layers are still being computed. ([Playbook, Section: data_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=data_parallelism))*

> **Playbook Quote for Figure 11:** "...gradients (red boxes) for a layer can be gathered and summed even before the gradients from earlier layers... have been computed... Overlapping computation and communication reduces the time spent waiting for gradient synchronization..." ([Playbook, Section: data_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=data_parallelism), adapted from figure and text).

The primary communication cost in DP stems from the AllReduce operation performed on the gradients in each step. Backpropagation computes gradients sequentially, starting from the final layers of the model. A naive approach might wait until all gradients are computed before initiating a single large AllReduce operation. This minimizes the number of communication calls but prevents any overlap between gradient computation and communication. Alternatively, one could initiate a separate AllReduce for each gradient tensor immediately as it becomes available. However, this results in numerous small communication calls, incurring significant overhead from network latency and potentially underutilizing network bandwidth. Gradient bucketing offers a more efficient approach. Instead of communicating immediately, computed gradients $\nabla\_w \ell(w, z)$ for various parameter blocks are grouped into larger, pre-defined buffers ("buckets"). An AllReduce operation is initiated only when a bucket is full. This strategy reduces the total number of communication calls compared to communicating each gradient individually, thus lowering latency overhead. It also allows for larger data transfers per call, potentially improving bandwidth utilization. Furthermore, while one bucket's gradients are being communicated via AllReduce, the backward pass can continue computing gradients for subsequent layers and filling the next bucket, enabling overlap between computation and communication. The total time cost of the AllReduce depends on the number of model parameters $\|w\|$ (determining the total data volume) and the specific implementation details, including bucketing strategy and the communication bandwidth between workers.

*(Instruction: Place Figure 12 from Playbook PDF here)*
![Figure 12: Data Parallelism with Gradient Bucketing](figures/dp_bucket.png)
*Figure 12: Data Parallelism with Gradient Bucketing. Gradients are grouped, and a single AllReduce is launched per bucket, enabling overlap. ([Playbook, Section: data_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=data_parallelism))*

> **Playbook Quote for Figure 12:** "...group gradients into buckets and launch a single all-reduce for all the gradients within the same bucket instead of performing independent all-reduce for each gradient." ([Playbook, Section: data_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=data_parallelism), adapted from figure caption).

Gradient accumulation is often used with data parallelism to simulate larger effective batch sizes without increasing the instantaneous memory load per worker. Let the desired global batch size be $\|\mathcal{B}\|$, distributed across $N_d$ workers. Each worker $k$ is initially assigned a portion of the data $\mathcal{B}\_k$, where $\cup\_{k=1}^{N_d} \mathcal{B}\_k = \mathcal{B}$ and $\|\mathcal{B}\_k\| = \|\mathcal{B}\|/N_d$. To perform gradient accumulation over $A$ steps, each worker further divides its data portion $\mathcal{B}\_k$ into $A$ smaller accumulation micro-batches $\mathcal{B}\_{k,1}, \dots, \mathcal{B}\_{k,A}$. For each accumulation step $a = 1, \dots, A$, worker $k$ performs a forward and backward pass using $\mathcal{B}\_{k,a}$, computing the local gradient

$$
g_{k,a} = \nabla_w \sum_{z \in \mathcal{B}_{k,a}} \ell(w, z).
$$

Instead of synchronizing immediately, worker $k$ accumulates these gradients locally, typically by summing them into a gradient buffer: $g\_k^{(A)} = \sum\_{a=1}^A g\_{k,a}.$ Only after all $A$ accumulation steps are complete on all workers is the synchronization performed. The accumulated local gradients $g\_k^{(A)}$ (which represent the gradient over the worker's entire portion $\mathcal{B}\_k$) are averaged across all $N\_d$ workers using a single AllReduce operation to compute the final average gradient

$$
\hat{g} = \frac{1}{N_d} \sum_{k=1}^{N_d} g_k^{(A)} = \frac{1}{|\mathcal{B}|} \sum_{k=1}^{N_d} \sum_{a=1}^A \sum_{z \in \mathcal{B}_{k,a}} \nabla_w \ell(w, z).
$$

A single optimizer step is then taken using this gradient $\hat{g}$. Crucially, gradient synchronization (AllReduce) must be disabled during the backward passes of the intermediate accumulation steps ($a=1, \dots, A-1$) and enabled only for the final step $A$. Frameworks often provide mechanisms, such as a `no_sync()` context manager in PyTorch's `DistributedDataParallel`, to manage this synchronization correctly, avoiding unnecessary communication overhead.

Data parallelism effectively parallelizes computation across the data dimension by distributing batches over $N_d$ workers. This can significantly increase training throughput (samples processed per second). However, its primary limitation remains memory consumption. Every worker must store the entire set of model parameters $w$, the corresponding optimizer states (e.g., momentum $m_t$ and variance $v_t$ buffers for Adam, denoted $\text{OptState}$), the full gradients $g_k$ during accumulation or before synchronization, and the activations $A\_{k,a}$ generated during the forward pass for its assigned micro-batch $\mathcal{B}\_{k,a}$. Consequently, DP alone is insufficient if the model's memory footprint (size of $w$ + size of $\text{OptState}$ + size of $A\_{k,a}$ for a minimal batch) exceeds the memory capacity of a single worker. Furthermore, the performance of DP relies on efficient gradient synchronization. The communication cost of the AllReduce operation, which scales with the model size $\|w\|$ (number of parameters), can become a bottleneck, especially as the number of workers $N_d$ increases or if the interconnect bandwidth between workers is limited.

### 6.2 ZeRO (Zero Redundancy Optimizer) Stages

The ZeRO techniques address the memory limitations of standard Data Parallelism (DP) by eliminating redundant storage of model state across the $N_d$ workers in the DP group. Instead of each worker replicating the optimizer states (e.g., the momentum and second moment buffers for Adam; denoted by $\text{OptState}$), gradients ($g$), and potentially parameters ($w$), ZeRO partitions these states, assigning each worker $k$ responsibility for only a $1/N_d$ fraction of the total state. Let $w = \cup_{k=1}^{N_d} w^{(k)}$, $g = \cup_{k=1}^{N_d} g^{(k)}$, and $\text{OptState} = \cup_{k=1}^{N_d} \text{OptState}^{(k)}$ represent this conceptual partitioning, where worker $k$ owns shard $(k)$. ([Playbook, Section: zero](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=zero)).

Activations $A_k$ depend on the local data $\mathcal{B}_k$ and cannot be sharded this way. However, optimizer states, gradients (after aggregation), and parameters are identical across replicas in standard DP and are thus candidates for partitioning. ZeRO implements this partitioning in three progressive stages.

**ZeRO Stage 1 (ZeRO-1)** partitions only the optimizer states. Each worker $k$ holds the full parameter set $w$ but only its shard of the optimizer state, $\text{OptState}^{(k)}$.
1.  Forward/Backward Pass: Each worker $k$ computes the full local gradient $g_k = \nabla_w \ell(w, \mathcal{B}_k)$.
2.  Gradient Synchronization/Sharding: A `ReduceScatter` collective operation is used. It sums the gradients $g_k$ across all workers and distributes the result such that worker $k$ receives only the gradient shard $\hat{g}^{(k)}$ corresponding to the parameters $w^{(k)}$ whose optimizer states $\text{OptState}^{(k)}$ it owns.
3.  Optimizer Step: Worker $k$ uses $\hat{g}^{(k)}$ and $\text{OptState}^{(k)}$ to update *only* its assigned parameter shard $w^{(k)}$ (often performing this update in higher precision, e.g., FP32).
4.  Parameter Synchronization: An `AllGather` collective is used to collect the updated parameter shards $w^{(k)}$ from all workers, reconstructing the complete, updated parameter set $w$ (typically in the working precision like BF16/FP16) on every worker for the next forward pass.

*(Instruction: Place Figure 13 from Playbook PDF here)*
![Figure 13: ZeRO Stage 1 shards optimizer states. Requires ReduceScatter for gradients and AllGather for parameters.](figures/zero1.png)
*Figure 13: ZeRO Stage 1 Communication Pattern. Optimizer states are sharded. Gradients are reduced and scattered, parameter updates happen locally, then parameters are gathered. ([Playbook, Section: zero](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=zero))*

> **Playbook Quote for Figure 13:** "The gradients $\Psi_g$ are not sharded across devices and are averaged via ReduceScatter instead of AllReduce. Each device ends up holding the gradients corresponding to its partition of the optimizer states... After the weights $\Psi_p$ have been updated... an AllGather operation is required to gather the weights onto all devices..." ([Playbook, Section: zero](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=zero), describing Figure 13).

**ZeRO Stage 2 (ZeRO-2)** partitions both optimizer states ($\text{OptState}^{(k)}$) and gradients. Worker $k$ still holds the full parameters $w$ temporarily during computation but only persists $\text{OptState}^{(k)}$ and its gradient shard $\hat{g}^{(k)}$.
1.  Forward Pass: Computes activations $A_k$.
2.  Backward Pass & Gradient Sharding: As gradients are computed, they are immediately reduced and scattered using `ReduceScatter`. Worker $k$ only stores the final, averaged gradient shard $\hat{g}^{(k)}$ relevant to its optimizer states. This avoids storing the full gradient $g_k$ temporarily, saving memory compared to ZeRO-1.
3.  Optimizer Step: Worker $k$ updates its parameter shard $w^{(k)}$ using $\hat{g}^{(k)}$ and $\text{OptState}^{(k)}$.
4.  Parameter Synchronization: An `AllGather` reconstructs the full parameters $w$ on all workers.

**ZeRO Stage 3 (ZeRO-3 / FSDP)** partitions all three components: parameters ($w^{(k)}$), gradients ($\hat{g}^{(k)}$), and optimizer states ($\text{OptState}^{(k)}$). Each worker $k$ *persistently* stores only its assigned shard $(k)$ of these states.
1.  Forward Pass: For each layer $j$ (or block of layers), workers perform an `AllGather` operation to temporarily collect the necessary full parameter tensor(s) $W_j$ for that layer. They compute the forward pass for that layer, $A_j = f_j(A_{j-1}; W_j)$, and then immediately discard the parameter shards of $W_j$ they do not own. Only the output activations $A_j$ and the owned parameter shard $w^{(k)}_j$ are kept.
2.  Backward Pass: Operates similarly. For layer $j$, the full parameters $W_j$ are gathered via `AllGather`. The backward pass computes the gradient, but only the portion corresponding to the locally owned parameter shard, $\nabla_{w^{(k)}_j} \ell$, is stored. These local gradient pieces are then synchronized and averaged across workers using `ReduceScatter` to obtain the final gradient shard $\hat{g}^{(k)}_j$.
3.  Optimizer Step: Worker $k$ updates its parameter shard $w^{(k)}$ using $\hat{g}^{(k)}$ and $\text{OptState}^{(k)}$. No final AllGather of parameters is needed, as they are gathered on-demand in the next forward pass.

*(Instruction: Place Figure 14 from Playbook PDF here)*
![Figure 14: ZeRO-3 / FSDP Forward (top) and Backward (bottom) parameter handling. Parameters are gathered on demand for computation and then released.](figures/dp_zero3.png)
*Figure 14: ZeRO-3 / FSDP Parameter Handling (Forward and Backward). Parameters are AllGathered before use and discarded afterward. Gradients are ReduceScattered. ([Playbook, Section: zero](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=zero))*

> **Playbook Quote for Figure 14:** "So as we perform the forward pass... we retrieve the necessary parameters on demand and immediately flush them... The backward pass works the same way..." ([Playbook, Section: zero](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=zero), adapted from figure descriptions).

The progressive partitioning offered by ZeRO stages provides substantial memory savings for model state compared to standard DP. ZeRO-1 saves memory on optimizer states, ZeRO-2 adds savings on gradients, and ZeRO-3 provides the maximum savings by also partitioning the parameters themselves.

*(Instruction: Place Figure 15 from Playbook PDF here)*
![Figure 15: ZeRO Memory Savings across stages. Shows reduction in Parameter ($\Psi_p$), Gradient ($\Psi_g$), and Optimizer State (k$\Psi_p$) memory per device vs DP degree $N_d$.](figures/zero_memory.png)
*Figure 15: ZeRO Memory Savings Overview. Illustrates theoretical memory reduction for parameters, gradients, and optimizer states per device for ZeRO stages 1, 2, and 3, compared to standard DP, as a function of DP group size $N_d$. ([Playbook, Section: zero](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=zero))*

> **Playbook Quote for Figure 15:** "Memory Consumption of ZeRO stages... $\Psi_p$ is the number of parameters... k depends on the choice of optimizer... $N_d$ is the data parallel degree." ([Playbook, Section: zero](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=zero), figure caption and legend).

The primary advantage of ZeRO is enabling data parallel training for models far too large to fit in a single worker's memory using standard DP. The main disadvantage is increased communication volume and complexity. ZeRO-1 and ZeRO-2 replace one gradient AllReduce with a ReduceScatter and add one parameter AllGather per step. ZeRO-3 replaces the single parameter AllGather with potentially many AllGather operations throughout the forward and backward passes (one per sharded parameter block being accessed). Efficient implementations rely on overlapping communication (prefetching parameters) with computation. Importantly, ZeRO partitions model state ($w, g, \text{OptState}$) but does *not* partition activation memory $A_k$.

### 6.3 Tensor Parallelism (TP)

Tensor Parallelism (TP) operates *within* individual layers, parallelizing the execution of large operations, most notably the matrix multiplications central to transformer MHA and FFN layers. It addresses cases where even a single layer's parameters or intermediate activations are too large for one device, or where computation within a layer needs to be accelerated. TP partitions the tensors and computations across a group of $N_{tp}$ workers. ([Playbook, Section: tensor_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism)).

Consider a linear layer computing $Y = f(XA)$ where $X \in \mathbb{R}^{b \times s \times h_{in}}$ is the input activation and $A \in \mathbb{R}^{h_{in} \times h_{out}}$ is the weight matrix ($f$ could be identity or include bias/activation). TP typically splits the weight matrix $A$ across the $N_{tp}$ workers:

*   **Column Parallelism:** Split $A$ column-wise: $A = [A_1 \| A_2 \| \dots \| A_{N_{tp}}]$, where each shard $A_k \in \mathbb{R}^{h_{in} \times (h_{out}/N_{tp})}$. The input $X$ must be fully available to all $N_{tp}$ workers. Each worker $k$ computes a corresponding shard of the output: $Y_k = f(X A_k) \in \mathbb{R}^{b \times s \times (h_{out}/N_{tp})}$. The full output $Y$ is represented by the concatenation $Y = [Y_1 \| \dots \| Y_{N_{tp}}]$ across the workers' memory. In the backward pass, computing the gradient w.r.t. $X$ involves summing contributions from all workers: $\nabla_X \ell = \sum_{k=1}^{N_{tp}} (\nabla_{Y_k} \ell) A_k^T$, typically requiring an `AllReduce` operation.

*   **Row Parallelism:** Split $A$ row-wise: $A = [A_1^T \| \dots \| A_{N_{tp}}^T]^T$, where $A_k \in \mathbb{R}^{(h_{in}/N_{tp}) \times h_{out}}$. The input $X$ must also be sharded row-wise (along the $h_{in}$ dimension): $X = [X_1 \| \dots \| X_{N_{tp}}]$, where $X_k \in \mathbb{R}^{b \times s \times (h_{in}/N_{tp})}$. This sharding often naturally results from a preceding column-parallel layer. Each worker $k$ computes a partial output using its input and weight shards: $Y_k = f(X_k A_k)$. The final output $Y \in \mathbb{R}^{b \times s \times h_{out}}$ is obtained by summing these partial outputs across all workers: $Y = \sum_{k=1}^{N_{tp}} Y_k$, which requires an `AllReduce` operation. The backward pass for $\nabla_X \ell$ can proceed locally on each worker using the full $\nabla_Y \ell$ (obtained via identity communication) and the local weight shard $A_k$.

*(Instruction: Place Figure 16 from Playbook PDF here)*
![Figure 16: Column (left) and Row (right) Parallelism for a Linear Layer $Y=XA$. Shows sharding of A and necessary communication for inputs/outputs.](figures/tp_linear.png)
*Figure 16: Tensor Parallelism for Linear Layers (Column and Row). Column parallelism splits A column-wise, requires full X, outputs sharded Y. Row parallelism splits A row-wise, requires sharded X, requires AllReduce to get full Y. ([Playbook, Section: tensor_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism))*

> **Playbook Quote for Figure 16:** [Find quote in playbook describing the TP diagram(s) for linear layers, e.g., "...column parallel approach splits the weight matrix A vertically... The input X needs to be broadcasted... row parallel approach splits the weight matrix A horizontally... requires an all-reduce operation on the output Y."] ([Playbook, Section: tensor_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism), adapt quotes for column/row figures if separate).

In transformer blocks, FFN layers are commonly parallelized using column parallelism for the first linear layer ($W_1$) followed by row parallelism for the second ($W_2$). This pattern, sometimes called MLP parallelism, avoids an AllReduce between the two layers. MHA layers use column parallelism for the $Q, K, V$ projections ($W_Q, W_K, W_V$) and row parallelism for the output projection ($W_O$). The number of attention heads $a$ must typically be divisible by the TP degree $N_{tp}$.

*(Instruction: Place Figure 17 from Playbook PDF here)*
![Figure 17: Applying Tensor Parallelism to Transformer FFN and MHA blocks.](figures/tp_transformer.png)
*Figure 17: Tensor Parallelism Applied to Transformer FFN and MHA. Shows typical column/row configuration for FFN and attention projections. ([Playbook, Section: tensor_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism))*

> **Playbook Quote for Figure 17:** [Find quote in playbook describing the application of column/row parallelism within FFN ("MLP") and MHA ("Attention") blocks.] ([Playbook, Section: tensor_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism)).

TP requires collective communication operations (like `AllReduce`) *within* the forward and backward computations of individual layers. This implies frequent synchronization and necessitates very high-bandwidth, low-latency interconnects between the participating workers, such as NVLink typically found within a single compute node.

*(Instruction: Place Figure 18 from Playbook PDF here)*
![Figure 18: Communication (AllReduce) introduces synchronization points within TP layer computation.](figures/tp_comm.png)
*Figure 18: Communication Timeline in Tensor Parallelism. AllReduce operations create synchronization points within layer execution, potentially leading to communication overhead. ([Playbook, Section: tensor_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism))*

> **Playbook Quote for Figure 18:** "...we hit a synchronization point with the AllReduce operation that cannot be overlapped with computation. This exposed communication overhead is necessary to combine partial results across tensor-parallel ranks..." ([Playbook, Section: tensor_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism)).

The main benefit of TP is reducing the memory footprint per worker for weights $w$, gradients $g$, optimizer states $\text{OptState}$, *and* activations $A$ associated with the parallelized layers, as these are sharded across the $N_{tp}$ workers. This allows fitting larger layers onto devices. TP can also reduce the wall-clock time per layer if computation savings dominate the added communication cost. However, the high communication overhead limits TP's scalability, making it most effective across a small number of tightly connected workers (e.g., $N_{tp} \le 8$). It also increases implementation complexity significantly. Furthermore, operations that require the full hidden dimension (like Layer Normalization) cannot be directly parallelized by TP along the hidden dimension, limiting overall activation memory savings unless combined with other techniques like Sequence Parallelism.

### 6.4 Sequence Parallelism (SP)

Sequence Parallelism (SP) is an optimization technique used in conjunction with Tensor Parallelism (TP) to further reduce activation memory. TP effectively shards tensors along the hidden dimension ($h$) for matrix multiplications but struggles with operations like Layer Normalization or Dropout that typically require the full activation vector for each token. SP addresses this by sharding activations along the *sequence* dimension ($s$) specifically for these non-TP-friendly operations, while using TP's hidden dimension sharding for the matrix multiplies. ([Playbook, Section: sequence_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=sequence_parallelism)).

Consider a group of $N_{tp}$ workers performing TP. With SP, the input activation tensor $X \in \mathbb{R}^{b \times s \times h}$ is initially sharded along the sequence dimension, so worker $k$ holds $X_k \in \mathbb{R}^{b \times (s/N_{tp}) \times h}$. Operations like LayerNorm can be applied locally to $X_k$. However, to perform a TP operation like a column-parallel linear layer ($Y_k = X W_k$), the full input $X$ along the sequence dimension is required. Thus, before the TP operation, an `AllGather` collective is performed across the TP workers along the sequence dimension to reconstruct $X$ on each worker. Conversely, after a TP operation like a row-parallel layer (which produces partial sums $Y_k$ that are summed via AllReduce to get $Y$), if the subsequent operation (e.g., LayerNorm) expects sequence-sharded input, a `ReduceScatter` collective is performed along the sequence dimension. This `ReduceScatter` simultaneously performs the summation needed by the row-parallel layer and distributes the final result $Y$ correctly sharded by sequence dimension $s$ across the workers.

*(Instruction: Place Figure 19 from Playbook PDF here)*
![Figure 19: Tensor sharding transitions between Sequence Parallel (SP) regions (e.g., for LayerNorm) and Tensor Parallel (TP) regions (e.g., for Linear layers) using AllGather and ReduceScatter along the sequence dimension.](figures/sp_comm.png)
*Figure 19: Tensor Sharding and Communication in TP vs. TP+SP. Shows transitions using AllGather (g) and ReduceScatter (g*) along the sequence dimension when moving between TP-sharded and SP-sharded regions. ([Playbook, Section: sequence_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=sequence_parallelism))*

> **Playbook Quote for Figure 19:** [Find quote in playbook explaining the figure with f/f*/g/g*, describing the transitions: "g" operation (all-gather) combines... back to full sequence length... "g*" operation (reduce-scatter) which reduces... while scattering along sequence dimension".] ([Playbook, Section: sequence_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=sequence_parallelism)).

The overall communication volume with SP is generally similar to that of TP alone, but the communication patterns change. Instead of potentially broadcasting inputs for column parallelism, SP uses `AllGather` along the sequence dimension. Instead of only using `AllReduce` for row parallelism outputs, SP uses `ReduceScatter` along the sequence dimension. These operations still require high-bandwidth intra-node interconnects.

The primary benefit of Sequence Parallelism is reducing the peak activation memory footprint beyond what TP alone achieves. By keeping activations sharded along the sequence dimension for operations like LayerNorm, it avoids materializing the full $b \times s \times h$ tensor for these steps. This can permit training with larger batch sizes or longer sequences within a TP group. However, SP inherits the communication bottlenecks and limited scalability (typically intra-node) of TP, while adding complexity due to the necessary tensor redistribution operations (AllGather, ReduceScatter) at the boundaries between SP and TP computations.

### 6.5 Context Parallelism (CP) & Ring Attention

Context Parallelism (CP) is a technique specifically designed to handle extremely long input sequences ($s$), where activation memory becomes prohibitive even with TP, SP, and activation recomputation. Unlike SP which shards specific operations along the sequence axis within a TP group, CP globally partitions the input sequence across a group of $N_{cp}$ workers for almost the entire computation. Worker $k$ holds a chunk of the sequence $X_k \in \mathbb{R}^{b \times (s/N_{cp}) \times h}$. ([Playbook, Section: context_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=context_parallelism)).

Most operations in a transformer block, such as the FFN layers and Layer Normalization, can be performed entirely locally on each worker's sequence chunk $X_k$ without requiring communication. The primary challenge lies in the self-attention mechanism (MHA), as calculating attention scores $\text{softmax}(\frac{Q K^T}{\sqrt{d_k}})$ requires interaction between queries ($Q$) from one chunk and keys ($K$) from all other chunks in the sequence.

Ring Attention provides an efficient communication pattern to compute global attention under CP. Workers are arranged in a logical ring. The computation proceeds in $N_{cp}-1$ steps. In each step:
1.  Each worker $k$ computes partial attention scores using its local queries $Q_k$ against the key/value chunks ($K_{curr}, Val_{curr}$) it currently holds.
2.  Simultaneously, worker $k$ sends its current $K_{curr}, Val_{curr}$ chunk to the next worker ($k+1 \pmod{N_{cp}}$) in the ring.
3.  Worker $k$ receives the key/value chunk ($K_{prev}, Val_{prev}$) from the previous worker ($k-1 \pmod{N_{cp}}$). These become $K_{curr}, Val_{curr}$ for the next step.
This process allows the communication of K/V chunks to be overlapped with the local attention score computation (specifically the $Q K^T$ matrix multiplication), hiding much of the communication latency.

*(Instruction: Place Figure 20 from Playbook PDF here)*
![Figure 20: Ring Attention mechanism for Context Parallelism. K/V chunks circulate around the ring, overlapping communication with local attention computation.](figures/ring_attention.png)
*Figure 20: Ring Attention Mechanism. Illustrates the step-by-step passing of K/V chunks around the ring of workers, enabling global attention calculation with overlapped communication. ([Playbook, Section: context_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=context_parallelism))*

> **Playbook Quote for Figure 20:** [Find quote in playbook describing the ring attention animation/diagram, explaining the passing of K/V chunks and overlap: "each GPU first initiates an asynchronous communication... computes the attention score... receives keys and values from the previous GPU..."] ([Playbook, Section: context_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=context_parallelism)).

For causal language models where tokens only attend to previous tokens, a simple sequential partitioning of the sequence for CP can lead to load imbalance in Ring Attention (workers holding early chunks have less computation). ZigZag attention partitioning distributes sequence chunks non-contiguously across workers to mitigate this and balance the computational load more evenly.

*(Instruction: Place Figure 21 from Playbook PDF here)*
![Figure 21: ZigZag partitioning assigns sequence chunks non-sequentially to workers to improve load balance in causal Ring Attention under Context Parallelism.](figures/cp_zigzagmask.png)
*Figure 21: ZigZag Partitioning for Load Balancing in Ring Attention. Shows how non-sequential chunk assignment balances computation across workers for causal masks. ([Playbook, Section: context_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=context_parallelism))*

> **Playbook Quote for Figure 21:** "...assigning the tokens not purely sequential to the GPUs but by mixing the ordering... computation is now balanced across all GPUs." ([Playbook, Section: context_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=context_parallelism), adapted from figure description).

Communication in CP involves the ring-based point-to-point exchange of K/V chunks during attention computation. Additionally, because each worker processes different parts of the sequence data, CP functions similarly to Data Parallelism regarding gradients. An `AllReduce` operation across the $N_{cp}$ workers is required after the backward pass to average the gradients before the optimizer step.

Context Parallelism, via Ring Attention, enables training models on exceptionally long sequences by keeping activation memory sharded along the sequence dimension for most of the model. It provides an efficient way to compute global attention with distributed sequences. However, it introduces complexity, particularly in the attention layer, and still requires gradient synchronization via AllReduce across the CP group.

### 6.6 Pipeline Parallelism (PP) & Scheduling

Pipeline Parallelism (PP) is primarily employed when the model parameters $\|w\|$ are too large to fit onto a single device or even a Tensor Parallelism group, or when scaling training across a large number of compute nodes where TP's high intra-node bandwidth assumption does not hold. PP partitions the model layers *sequentially* into $P$ stages. Each stage $p \in \{1, \dots, P\}$, consisting of a contiguous block of layers with parameters $w_p$, is assigned to a different worker or group of workers (forming a pipeline group of size $N_p = P$). Data flows sequentially through the pipeline: the output activations $A_p$ of stage $p$ become the input $A\_{p}$ for stage $p+1$. Formally, $A\_p = f\_p(A\_{p-1}; w_p)$, where $f_p$ represents the computation of stage $p$. ([Playbook, Section: pipeline_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=pipeline_parallelism)).

Executing this sequentially for a full data batch $\mathcal{B}$ (i.e., computing $A\_1 = f\_1(\mathcal{B})$, then $A\_2 = f\_2(A\_1)$, ..., then performing the full backward pass) is highly inefficient. At any moment, only one stage $p$ is active (either computing $f\_p$ or its backward counterpart $b\_p$), leaving the other $P-1$ stages idle. This idle time is referred to as the "pipeline bubble." The fraction of idle time increases with the number of stages $P$, significantly hindering hardware utilization.

*(Instruction: Place Figure 22 from Playbook PDF here)*
![Figure 22: Pipeline Parallelism with naive scheduling leads to a large "bubble" of idle time (grey areas).](figures/pp_bubble.png)
*Figure 22: Pipeline Bubble in Naive Sequential Execution. Only one stage is active at a time, resulting in significant idle periods (bubble). ([Playbook, Section: pipeline_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=pipeline_parallelism))*

> **Playbook Quote for Figure 22:** "An example of Pipeline parallelism... The remaining idle time is indicated in grey and usually called the 'bubble'..." ([Playbook, Section: pipeline_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=pipeline_parallelism)).

To reduce the pipeline bubble, the global data batch $\mathcal{B}$ is split into $m$ smaller micro-batches, $\mathcal{B}^{(1)}, \dots, \mathcal{B}^{(m)}$. The pipeline processes these micro-batches in a staggered fashion, allowing different stages to work on different micro-batches concurrently. The specific order of forward ($F$) and backward ($B$) passes for micro-batches defines the pipeline schedule. Common schedules include:

*   **All-Forward-All-Backward (AFAB) / GPipe:** All micro-batches complete their forward pass through all $P$ stages ($F_1^{(j)}, F_2^{(j)}, \dots, F_P^{(j)}$ for $j=1..m$). Only then does the backward pass begin for all micro-batches ($B_P^{(j)}, B_{P-1}^{(j)}, \dots, B_1^{(j)}$ for $j=1..m$). This schedule is relatively simple but requires storing the activations $A_{p-1}^{(j)}$ for *all* micro-batches $j$ at each stage boundary $p$ until the corresponding backward pass $B_p^{(j)}$ is executed. Peak activation memory is roughly proportional to $m$. The relative size of the bubble (fraction of idle time) is reduced to approximately $(P-1)/m$.

*   **One-Forward-One-Backward (1F1B) / Interleaved:** This schedule interleaves forward and backward passes more tightly to reduce activation memory. For example, after micro-batch $j$ completes its forward pass $F_P^{(j)}$, its backward pass $B_P^{(j)}, \dots, B_1^{(j)}$ can begin immediately, potentially overlapping with the forward passes $F_1^{(j+k)}, \dots, F_P^{(j+k)}$ of subsequent micro-batches. This significantly reduces the number of activations that need to be stored simultaneously, with peak activation memory roughly proportional to the number of stages $P$ rather than the number of micro-batches $m$. The bubble size remains approximately $(P-1)/m$, but implementation complexity increases.

*(Instruction: Place Figure 23 from Playbook PDF here)*
![Figure 23: Pipeline schedules: All-Forward-All-Backward (GPipe-like, left) vs. One-Forward-One-Backward (Interleaved, right). 1F1B reduces activation memory compared to AFAB.](figures/pp_schedules.png)
*Figure 23: Pipeline Parallelism Schedules (AFAB and 1F1B). AFAB (top/left in source) has high activation memory. 1F1B (bottom/right in source) interleaves passes to reduce activation memory. Both have similar bubble sizes. ([Playbook, Section: pipeline_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=pipeline_parallelism))*

> **Playbook Quote for Figure 23:** [Find quotes describing the AFAB (`pp_afab2.svg`) and 1F1B (`image.png` with 1F1B schedule) figures, highlighting micro-batch flow and activation storage needs, e.g., "GPipe (All Forward All Backward)... stores the activations for all micro-batches", "1F1B... reduces the memory footprint... by performing backward passes as soon as possible"] ([Playbook, Section: pipeline_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=pipeline_parallelism)).

More advanced schedules aim to further reduce or eliminate the bubble. **Interleaved Stages** (not the same as 1F1B scheduling) assign non-contiguous layers to pipeline stages (e.g., layers 1,3,5 on stage 1; layers 2,4,6 on stage 2) which can sometimes reduce dependencies and bubble size but increases inter-stage communication. Techniques like **ZeroBubble** or **DualPipe** use fine-grained analysis of backward pass dependencies (separating weight gradient calculation $\nabla w_p$ from input gradient calculation $\nabla A\_{p-1}$) to construct complex schedules that achieve near-zero bubble size, often at the cost of significantly increased implementation difficulty.

Communication in PP mainly involves point-to-point transfers of activation tensors $A\_p^{(j)}$ forward and gradient tensors $\nabla A\_p^{(j)}$ backward between adjacent pipeline stages ($p$ and $p+1$). The size of these transfers depends on the batch size, sequence length, hidden dimension, and the micro-batch size $m$. This communication pattern is generally less demanding on network bandwidth compared to the large collective operations in TP or FSDP, making PP suitable for scaling across multiple compute nodes with standard network interconnects.

Pipeline Parallelism excels at partitioning very deep models across many devices, enabling the training of models whose total parameter count $\|w\|$ is extremely large. It is generally less sensitive to inter-node communication bandwidth than TP or FSDP. However, the pipeline bubble inherently limits perfect scaling efficiency. Achieving good performance requires careful load balancing (ensuring each stage $p$ has similar computational cost) and selecting an appropriate schedule (trading off activation memory, bubble size, and complexity).

### 6.7 Expert Parallelism (EP)

Expert Parallelism (EP) is a specialized technique applicable only to Mixture-of-Experts (MoE) models. MoE architectures modify standard transformer blocks by replacing certain layers, typically the FFN, with a collection of $E$ parallel "expert" networks (usually FFNs themselves), $\{f_{expert, e}\}_{e=1}^E$. For each input token representation $x_t$, a lightweight gating network $R(x_t)$ dynamically selects a small subset of these experts (often just the top 1 or top 2) to process that specific token. ([Playbook, Section: expert_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=expert_parallelism)).

*(Instruction: Place Figure 24 from Playbook PDF here)*
![Figure 24: Conceptual illustration of a Mixture-of-Experts (MoE) layer, where a router selects specific experts (e.g., FFNs) for each token.](figures/moe_layer.png)
*Figure 24: Mixture-of-Experts (MoE) Layer Concept. A gating network routes each token to one or a few expert networks. ([Playbook, Section: expert_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=expert_parallelism))*

> **Playbook Quote for Figure 24:** "Illustrationg [sic] of a MoE layer taken from the Switch Transformers paper" ([Playbook, Section: expert_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=expert_parallelism), figure caption).

Expert Parallelism distributes these $E$ expert networks across a group of $N_{ep}$ workers. Worker $k$ might hold the parameters $w_e$ for a subset of experts $E_k$. Since only a few experts process each token, EP leverages this sparsity. During the forward pass, after the gating network $R$ determines the expert assignment $e(x_t)$ for each token representation $x_t$, an `AlltoAll` collective communication operation is performed. This operation efficiently routes each token $x_t$ from its current worker to the specific worker(s) responsible for holding the assigned expert(s) $e(x_t)$. The receiving worker $k$ then computes the expert output $y_t = f_{expert, e(x_t)}(x_t; w_{e(x_t)})$. A second `AlltoAll` operation is used to gather the computed outputs $y_t$ back to their original workers, combining them according to the gating weights. A similar pattern occurs in the backward pass.

Expert Parallelism by itself only parallelizes the MoE layers. Other model components (like attention layers or non-MoE FFNs) would still be replicated and perform redundant computations across the $N_{ep}$ workers. Therefore, EP is almost always used in combination with Data Parallelism (DP). The workers are arranged conceptually in a 2D grid of size $N_d \times N_{ep}$. Within each row (fixed DP rank), workers perform EP, distributing the experts. Within each column (fixed EP rank), workers perform DP, replicating the non-expert parts of the model and the expert parameters assigned to that column, while processing different slices of the data batch. Gradient synchronization for the non-expert parameters occurs via AllReduce within each EP group (column), while token routing for experts occurs via AlltoAll within each DP group (row).

*(Instruction: Place Figure 25 from Playbook PDF here)*
![Figure 25: Combining Expert Parallelism (EP) and Data Parallelism (DP). Experts are sharded across EP ranks (columns), while non-expert layers and data are handled by DP ranks (rows).](figures/ep_dp_combo.png)
*Figure 25: Combining Expert Parallelism (EP) and Data Parallelism (DP). Experts are distributed across one dimension of the worker grid, while data parallelism operates along the other. ([Playbook, Section: expert_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=expert_parallelism))*

> **Playbook Quote for Figure 25:** [Find quote in playbook describing the `ep_schema.png` figure, illustrating the combination of DP and EP, likely showing AllReduce within EP groups and AlltoAll within DP groups.] ([Playbook, Section: expert_parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=expert_parallelism)).

Expert Parallelism enables scaling models to extremely large parameter counts (by increasing the number of experts $E$) while keeping the computational cost (FLOPs) per token relatively low, as each token only passes through a small number of experts. The main challenge is the communication overhead associated with the `AlltoAll` operations required for routing tokens, which can be significant, especially as the number of experts and workers increases. EP is inherently tied to the MoE architecture.

## 7. Combining Strategies & Conclusion

The parallelism strategies discussed in Section 6—Data Parallelism (DP/ZeRO), Pipeline Parallelism (PP), Tensor Parallelism (TP/SP/CP), and Expert Parallelism (EP)—each address specific bottlenecks in large-scale training. However, each also has limitations. Standard DP is constrained by single-device memory; ZeRO increases communication volume; TP scalability is typically limited by intra-node bandwidth; PP introduces pipeline bubbles; CP adds overhead for long sequences; EP applies only to MoE models. Consequently, training the largest state-of-the-art models almost always requires combining multiple parallelism strategies to leverage their complementary strengths while mitigating their individual weaknesses ([Playbook, Section: 5d_parallelism_in_a_nutshell](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=5d_parallelism_in_a_nutshell)).

A common and foundational approach is **3D Parallelism**, which combines Data Parallelism ($N_d$ workers), Pipeline Parallelism ($P$ stages), and Tensor Parallelism ($N_{tp}$ workers). The total number of workers is $N = N_d \times P \times N_{tp}$. Conceptually, these workers are arranged in a 3D grid. The typical mapping leverages hardware topology: TP is used *within* a compute node, utilizing high-speed interconnects like NVLink among its $N_{tp}$ workers to handle the frequent communication required for intra-layer parallelism. PP is used *across* nodes, partitioning the model's layers sequentially into $P$ stages, relying on the less frequent inter-stage communication (activations forward, gradients backward) which can tolerate slower inter-node network bandwidth. DP then replicates this entire PP+TP structure $N_d$ times, processing different data batches on each replica, with gradient synchronization occurring across these replicas.

In modern large-scale training, the Data Parallelism dimension ($N_d$) within combined strategies like 3D parallelism is often implemented using ZeRO, particularly ZeRO-3/FSDP, rather than standard DP. The primary benefit is memory efficiency. By sharding the model parameters $w$, gradients $g$, and optimizer states $\text{OptState}$ across the $N_d$ workers (Section 6.2), FSDP drastically reduces the memory required per worker compared to replicating the full model state. This allows the combined system (e.g., FSDP + PP + TP) to accommodate significantly larger models or use larger per-replica batch sizes than would be possible with standard DP + PP + TP. While simpler ZeRO stages like ZeRO-1 or ZeRO-2 can also be combined with PP to save optimizer state or gradient memory respectively, FSDP offers the most substantial savings by partitioning the parameters themselves.

When considering how to partition large model parameters across devices, Pipeline Parallelism (PP) and ZeRO-3 (FSDP) offer different approaches, as highlighted in the playbook ([Playbook, Section: 5d_parallelism_in_a_nutshell](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=5d_parallelism_in_a_nutshell)). PP partitions the model *layer-wise* into sequential stages; worker $p$ holds the full parameters $w_p$ for the layers in its stage. ZeRO-3 partitions *parameter-wise* across DP workers; worker $k$ holds shards $w^{(k)}$ of parameters from *all* layers. This leads to different communication patterns: PP primarily communicates activations $A_p$ and activation gradients $\nabla A_p$ between stages via point-to-point messages, whereas ZeRO-3 communicates parameter shards $w^{(k)}$ via `AllGather` collectives within the forward/backward pass. Consequently, their performance sensitivities differ: PP's efficiency is heavily influenced by the pipeline bubble, which can be mitigated by increasing the number of micro-batches (often related to gradient accumulation `grad_acc`); ZeRO-3's efficiency relies on overlapping the parameter `AllGather` communication with computation, which typically benefits from larger micro-batch sizes (`mbs`) or sequence lengths (`s`). While combining PP and ZeRO-3 is technically possible, the playbook notes it can be complex and may demand very large global batch sizes to amortize the communication overheads effectively.

Context Parallelism (CP) and Expert Parallelism (EP) are typically viewed as orthogonal additions, layered onto a base parallelism configuration (like FSDP + PP + TP) to address specific needs. CP is employed when dealing with extremely long sequences ($s$), adding sequence-dimension sharding and Ring Attention communication, usually across the DP or PP dimension workers. EP is used exclusively for MoE models, distributing the expert networks across a dimension of workers (often the DP dimension) and introducing `AlltoAll` communication for token routing.

The interplay between these strategies allows for flexible configurations tailored to specific model sizes and hardware capabilities. The following diagram and table summarize the core ideas and trade-offs.

*(Instruction: Place Figure 26 from Playbook PDF - likely `5d_parallelism_combo.svg` - here)*
![Figure 26: Conceptual view combining different parallelism dimensions (DP/ZeRO, PP, TP, EP, CP) acting on model weights, activations, and data.](figures/5d_parallelism_combo.png)
*Figure 26: Conceptual Overview of Combined Parallelism Dimensions. Illustrates how different strategies partition activations, weights, and data along different axes. ([Playbook, Section: 5d_parallelism_in_a_nutshell](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=5d_parallelism_in_a_nutshell))*

> **Playbook Quote for Figure 26:** [Find quote in playbook describing the composite diagram, e.g., summarizing the different sharding dimensions.] ([Playbook, Section: 5d_parallelism_in_a_nutshell](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=5d_parallelism_in_a_nutshell)).

*(Instruction: Place Table 1 from Playbook PDF - summarizing strategies - here)*
**Table 1: Summary of Parallelism Strategies**
*(Reproduce or reference the summary table from the playbook comparing Method, Memory Savings, Parallel Dimension, and Disadvantage)*
> **Playbook Quote for Table 1:** [Find quote in playbook introducing or summarizing the comparison table.] ([Playbook, Section: 5d_parallelism_in_a_nutshell](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=5d_parallelism_in_a_nutshell)).

Choosing the optimal combination and configuration involves a structured process, as outlined in the playbook ([Playbook, Section: finding_the_best_training_configuration](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=finding_the_best_training_configuration)). The general approach involves iterative refinement: first, ensuring a single training step fits in memory using necessary parameter partitioning (PP, TP, ZeRO-3) and activation management (recomputation); second, scaling to the target global batch size using data-centric parallelism (DP/FSDP, CP) and gradient accumulation; and third, optimizing for throughput by tuning the degrees of parallelism (e.g., maximizing intra-node TP) and micro-batch sizes while respecting communication bottlenecks. The best strategy is highly dependent on the specific model architecture, the number and type of available accelerators, and the characteristics of the network interconnects.

In conclusion, training large-scale transformer models pushes the boundaries of computation and memory available on single devices. Addressing this requires moving beyond individual optimizers to system-level parallelism strategies. We have reviewed the transformer architecture's demands, the critical bottleneck of activation memory, and the utility of activation recomputation. We explored the primary parallelism techniques detailed in the Ultrascale Playbook: Data Parallelism (enhanced by ZeRO/FSDP), Pipeline Parallelism, Tensor Parallelism (enhanced by SP), Context Parallelism, and Expert Parallelism. Successful training at the largest scales necessitates thoughtfully combining these techniques—often in a 3D (DP/FSDP + PP + TP) or more complex arrangement—to optimally balance computation, memory usage, and communication overhead for the specific model and hardware infrastructure. Resources like the Ultrascale Playbook provide invaluable practical guidance for navigating these complex trade-offs.