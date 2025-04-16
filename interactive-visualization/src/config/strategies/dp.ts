import type { StepDetail, CommOperation, CommDataType } from '../context/types';
const MODEL_LAYERS = ['Embed', 'MHA', 'FFN', 'LN', 'Output'];
export const generateDpSteps = (numGpus: number): StepDetail[] => {
    const N = numGpus; const steps: StepDetail[] = [];
    steps.push({ step: 0, type: 'INIT', description: 'DP Init: Model $w$ replicated.', notation: `w_k = w`, strategy: 'dp' });
    MODEL_LAYERS.forEach((layer, index) => {
        const prevLayerAct = index > 0 ? MODEL_LAYERS[index-1] : 'Input';
        steps.push({ step: index + 1, type: 'COMPUTE', direction: 'forward', parallel: true, layer: layer, description: `Parallel Fwd Pass: Proc. ${layer} (Batch $B_k$).`, notation: `A_{${layer},k} = \\text{Forward}(A_{${prevLayerAct},k}, w)`, strategy: 'dp', activationProduced: layer });
    });
    const forwardEndStep = MODEL_LAYERS.length + 1;
    // Backwards pass steps are simplified here. Activation consumption linking happens in context.
    steps.push({ step: forwardEndStep, type: 'GRADIENTS', parallel: true, layer: `Grads`, description: `Backward Complete. Compute local $g_k$ from $B_k$.`, notation: `g_k = \\nabla_w \\mathcal{L}(B_k, w)`, strategy: 'dp' });
    steps.push({ step: forwardEndStep + 1, type: 'COMM', parallel: true, operation: 'AllReduce' as CommOperation, dataType: 'Gradients' as CommDataType, description: `Communication: AllReduce gradients $g_k$.`, notation: `\\hat{g} = \\frac{1}{${N}} \\sum_{k=0}^{${N}-1} g_k`, strategy: 'dp' });
    steps.push({ step: forwardEndStep + 2, type: 'UPDATE', parallel: true, layer: 'Optimizer', description: `Optimizer Step: Update replicas with $\\hat{g}$.`, notation: `w \\leftarrow \\text{OptimizerStep}(w, \\hat{g})`, strategy: 'dp' });
    steps.push({ step: forwardEndStep + 3, type: 'DONE', description: `DP Step Complete.`, strategy: 'dp' });
    return steps.map((s, index) => ({ ...s, step: index }));
};
