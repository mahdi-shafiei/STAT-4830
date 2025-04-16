import type { StepDetail, CommOperation, CommDataType } from '../context/types';
const MODEL_LAYERS = ['Embed', 'MHA', 'FFN', 'LN', 'Output'];
export const generateFsdpSteps = (numGpus: number): StepDetail[] => {
    const N = numGpus; const steps: StepDetail[] = [];
    steps.push({ step: 0, type: 'INIT', description: 'FSDP Init: Sharded State.', notation: `\\text{State}_k = \\{ w^{(k)}, Opt^{(k)} \\}`, strategy: 'fsdp' });
    MODEL_LAYERS.forEach((layer, index) => { // --- Forward Pass ---
        const stepBase = steps.length; const prevLayer = index > 0 ? MODEL_LAYERS[index-1] : 'Input';
        steps.push({ step: stepBase, type: 'COMM', parallel: true, layer: layer, operation: 'AllGather' as CommOperation, dataType: 'Params' as CommDataType, description: `Fwd-${layer}: AllGather Params $w_{${layer}}^{(j)}$.`, notation: `W_{${layer}} = \\text{AllGather}_{j=0}^{N-1}(w_{${layer}}^{(j)})`, strategy: 'fsdp' });
        steps.push({ step: stepBase + 1, type: 'COMPUTE', direction: 'forward', parallel: true, layer: layer, description: `Fwd-${layer}: Compute Output $A_{${layer},k}$ (Batch $B_k$).`, notation: `A_{${layer},k} = \\text{Forward}(A_{${prevLayer},k}, W_{${layer}})`, strategy: 'fsdp', activationProduced: layer }); // Mark activation
        steps.push({ step: stepBase + 2, type: 'MEMORY_OP', parallel: true, layer: layer, operation: 'DiscardParams', description: `Fwd-${layer}: Discard non-owned $W_${layer}$ shards, keep $w_{${layer}}^{(k)}$.`, strategy: 'fsdp' });
    });
    [...MODEL_LAYERS].reverse().forEach((layer, index) => { // --- Backward Pass ---
        const stepBase = steps.length; const actLayerIndex = MODEL_LAYERS.length - 1 - index - 1; const actLayerName = actLayerIndex >= 0 ? MODEL_LAYERS[actLayerIndex] : 'Input'; const actNotation = actLayerIndex >= 0 ? `A_{${actLayerName},k}` : `\\text{Input}(B_k)`; const incomingGradNotation = `\\mathrm{d}A_{${layer}}`;
        steps.push({ step: stepBase, type: 'COMM', parallel: true, layer: layer, operation: 'AllGather' as CommOperation, dataType: 'Params' as CommDataType, description: `Bwd-${layer}: AllGather Params $w_{${layer}}^{(j)}$.`, notation: `W_{${layer}} = \\text{AllGather}_{j=0}^{N-1}(w_{${layer}}^{(j)})`, strategy: 'fsdp' });
        steps.push({ step: stepBase + 1, type: 'COMPUTE', direction: 'backward', parallel: true, layer: layer, description: `Bwd-${layer}: Compute local gradient $\\nabla W_{${layer}}$.`, notation: `\\nabla W_{${layer}} = \\text{ComputeGrad}(${incomingGradNotation}, ${actNotation}, W_{${layer}})`, strategy: 'fsdp', activationConsumedLayer: actLayerName }); // Mark activation consumed
        steps.push({ step: stepBase + 2, type: 'MEMORY_OP', parallel: true, layer: layer, operation: 'DiscardParams', description: `Bwd-${layer}: Discard non-owned $W_{${layer}}$ shards.`, strategy: 'fsdp' });
        steps.push({ step: stepBase + 3, type: 'COMM', parallel: true, layer: layer, operation: 'ReduceScatter' as CommOperation, dataType: 'Gradients' as CommDataType, description: `Bwd-${layer}: ReduceScatter Gradients.`, notation: `\\hat{g}_{${layer}}^{(k)} = \\text{ReduceScatter}_{j=0}^{N-1}(\\nabla W_{${layer}}^{(j)})`, strategy: 'fsdp' });
    });
    const updateStep = steps.length;
    steps.push({ step: updateStep, type: 'UPDATE', parallel: true, layer: 'Optimizer', description: `Optimizer Step: Update local shard $w^{(k)}$.`, notation: `w^{(k)} \\leftarrow \\text{OptimizerStep}(w^{(k)}, \\hat{g}^{(k)}, Opt^{(k)})`, strategy: 'fsdp' });
    steps.push({ step: updateStep + 1, type: 'DONE', description: `FSDP Step Complete.`, strategy: 'fsdp' });
    return steps.map((s, index) => ({ ...s, step: index }));
};
