import type { StepDetail } from './types';
const MODEL_LAYERS = ['Embed', 'MHA', 'FFN', 'LN', 'Output'];
export const generateSingleGpuSteps = (): StepDetail[] => {
    const steps: StepDetail[] = [];
    steps.push({ step: 0, type: 'INIT', description: 'Single GPU Initial state.', strategy: 'single' });
    MODEL_LAYERS.forEach((layer, index) => {
        const prevLayerAct = index > 0 ? `A_{${MODEL_LAYERS[index-1]}}` : `Input`;
        steps.push({ step: index + 1, type: 'COMPUTE', direction: 'forward', layer: layer, description: `Forward Pass: Processing ${layer} layer.`, notation: `A_{${layer}} = \\text{Forward}(${prevLayerAct}, w)`, strategy: 'single', activationProduced: layer });
    });
    // Conceptual Backward pass steps to show activation consumption
    [...MODEL_LAYERS].reverse().forEach((layer, index) => {
         const actLayerIndex = MODEL_LAYERS.length - 1 - index - 1;
         const actLayerName = actLayerIndex >= 0 ? MODEL_LAYERS[actLayerIndex] : 'Input';
         steps.push({ step: MODEL_LAYERS.length + 1 + index, type: 'COMPUTE', direction: 'backward', layer: layer, description: `Backward Pass: Compute Grads for ${layer} (uses Act: ${actLayerName}).`, strategy: 'single', activationConsumedLayer: actLayerName });
    });
    steps.push({ step: (MODEL_LAYERS.length * 2) + 1, type: 'DONE', description: `Pass Complete.`, strategy: 'single' });
    return steps.map((s, index) => ({ ...s, step: index }));
};
