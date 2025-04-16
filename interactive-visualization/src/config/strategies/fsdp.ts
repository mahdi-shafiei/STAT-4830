import type { StepDetail, CommOperation, CommDataType } from '../context/types';

const MODEL_LAYERS = ['Embed', 'MHA', 'FFN', 'LN', 'Output'];

export const generateFsdpSteps = (numGpus: number): StepDetail[] => {
    const N = numGpus;
    const steps: StepDetail[] = [];
    steps.push({ step: 0, type: 'INIT', description: 'FSDP Init: Sharded State.', notation: `\\text{State}_k = \\{ w^{(k)}, Opt^{(k)} \\}`, strategy: 'fsdp' });

    // --- Forward Pass ---
    MODEL_LAYERS.forEach((layer, index) => {
        const stepBase = steps.length;
        const prevLayerAct = index > 0 ? `A_{${MODEL_LAYERS[index-1]},k}` : '\\text{Input}_k'; // Use A_k notation
        // Step 1: AllGather
        steps.push({ step: stepBase, type: 'COMM', parallel: true, layer: layer, operation: 'AllGather' as CommOperation, dataType: 'Params' as CommDataType, description: `Fwd-${layer}: AllGather Params $w_{${layer}}^{(j)}$.`, notation: `W_{${layer}} = \\text{AllGather}_{j=0}^{N-1}(w_{${layer}}^{(j)})`, strategy: 'fsdp' });
        // Step 2: Compute
        steps.push({ step: stepBase + 1, type: 'COMPUTE', direction: 'forward', parallel: true, layer: layer, description: `Fwd-${layer}: Compute Output (Batch $B_k$).`, notation: `A_{${layer},k} = \\text{ForwardLayer}(${prevLayerAct}, W_{${layer}})`, strategy: 'fsdp' });
        // Step 3: Discard
        steps.push({ step: stepBase + 2, type: 'MEMORY_OP', parallel: true, layer: layer, operation: 'DiscardParams', description: `Fwd-${layer}: Discard non-owned $W_${layer}$ shards, keep $w_{${layer}}^{(k)}$.`, strategy: 'fsdp' }); // Clarified keep
    });

    // --- Backward Pass ---
    [...MODEL_LAYERS].reverse().forEach((layer, index) => {
        const stepBase = steps.length;
        const activationLayerIndex = MODEL_LAYERS.length - 1 - index - 1;
        const activationLayerName = activationLayerIndex >= 0 ? MODEL_LAYERS[activationLayerIndex] : 'Input'; // Activation used
        const activationNotation = activationLayerIndex >= 0 ? `A_{${activationLayerName},k}` : '\\text{Input}_k';
        const incomingGradNotation = `\\text{d}A_{${layer}}`; // Incoming gradient for activation
        // Step 1: AllGather
        steps.push({ step: stepBase, type: 'COMM', parallel: true, layer: layer, operation: 'AllGather' as CommOperation, dataType: 'Params' as CommDataType, description: `Bwd-${layer}: AllGather Params $w_{${layer}}^{(j)}$.`, notation: `W_{${layer}} = \\text{AllGather}_{j=0}^{N-1}(w_{${layer}}^{(j)})`, strategy: 'fsdp' });
        // Step 2: Compute Gradient
        steps.push({ step: stepBase + 1, type: 'COMPUTE', direction: 'backward', parallel: true, layer: layer, description: `Bwd-${layer}: Compute full local gradient $\\nabla W_{${layer}}$.`, notation: `\\nabla W_{${layer}} = \\text{ComputeGrad}(${incomingGradNotation}, ${activationNotation}, W_{${layer}})`, strategy: 'fsdp' }); // Added incoming grad dA
        // Step 3: Discard Params
        steps.push({ step: stepBase + 2, type: 'MEMORY_OP', parallel: true, layer: layer, operation: 'DiscardParams', description: `Bwd-${layer}: Discard non-owned $W_{${layer}}$ shards.`, strategy: 'fsdp' });
        // Step 4: ReduceScatter Gradients
        steps.push({ step: stepBase + 3, type: 'COMM', parallel: true, layer: layer, operation: 'ReduceScatter' as CommOperation, dataType: 'Gradients' as CommDataType, description: `Bwd-${layer}: ReduceScatter Gradients.`, notation: `\\hat{g}_{${layer}}^{(k)} = \\text{ReduceScatter}_{j=0}^{N-1}(\\nabla W_{${layer}}^{(j)})`, strategy: 'fsdp' }); // Indicate sum is implicit
    });

    const updateStep = steps.length;
    steps.push({ step: updateStep, type: 'UPDATE', parallel: true, layer: 'Optimizer', description: `Optimizer Step: Update local shard $w^{(k)}$.`, notation: `w^{(k)} \\leftarrow \\text{OptimizerStep}(w^{(k)}, \\hat{g}^{(k)}, Opt^{(k)})`, strategy: 'fsdp' });

    steps.push({ step: updateStep + 1, type: 'DONE', description: `FSDP Step Complete.`, strategy: 'fsdp' });

    return steps.map((s, index) => ({ ...s, step: index }));
};
