import type { StepDetail, CommOperation, CommDataType } from '../../context/types';

// Re-declare or import MODEL_LAYERS if not globally available
const MODEL_LAYERS = ['Embed', 'MHA', 'FFN', 'LN', 'Output']; // Consistent layer list
const N_tp = 2; // Fixed TP size for this chunk

export const generateTpSteps = (): StepDetail[] => {
    const steps: StepDetail[] = [];
    const Ntp = N_tp; // Use constant

    // Step 0: Init
    steps.push({ step: 0, type: 'INIT', description: `TP (N=${Ntp}) Init: Layer params sharded (conceptual).`, notation: `w_{\\text{Layer}} = [w_{\\text{Layer}}^{(0)} | w_{\\text{Layer}}^{(1)}]`, strategy: 'tp' });

    // --- Forward Pass ---
    let currentStep = 0; // Will be incremented before first use
    MODEL_LAYERS.forEach((layer, index) => {
        currentStep++;
        const prevLayerAct = index > 0 ? `A_{${MODEL_LAYERS[index-1]}}` : `\\text{Input}`; // Simplified notation for viz
        let activationProducedLayer: string | null = layer; // Assume layer produces activation by default

        switch (layer) {
            case 'MHA':
                // 1. QKV Proj (Column Parallel)
                steps.push({ step: currentStep, type: 'COMPUTE', direction: 'forward', parallel: true, layer: layer, subStep: 'QKV (Col)', description: `TP Fwd: Compute sharded Q_k, K_k, V_k from ${prevLayerAct}.`, notation: `Q_k, K_k, V_k = \\text{ColParallel}(${prevLayerAct}, W_{QKV})`, strategy: 'tp' });
                // 2. Attention Compute (Local/Head-sharded)
                currentStep++;
                steps.push({ step: currentStep, type: 'COMPUTE', direction: 'forward', parallel: true, layer: layer, subStep: 'Attn Comp', description: `TP Fwd: Compute local Attention Output O_k.`, notation: `O_k = \\text{Attention}(Q_k, K_k, V_k)`, strategy: 'tp' });
                // 3. Output Proj (Row Parallel)
                currentStep++;
                steps.push({ step: currentStep, type: 'COMPUTE', direction: 'forward', parallel: true, layer: layer, subStep: 'Out (Row)', description: `TP Fwd: Compute partial Output Projection Z_k.`, notation: `Z_k = \\text{RowParallel}(O_k, W_{Out})`, strategy: 'tp' });
                // 4. AllReduce Output Proj
                currentStep++;
                steps.push({ step: currentStep, type: 'COMM', parallel: true, layer: layer, operation: 'AllReduce' as CommOperation, dataType: 'Activations' as CommDataType, subStep: 'Out Reduce', description: `TP Fwd: AllReduce partial outputs Z_k.`, notation: `A_{${layer}} = \\text{AllReduce}_k(Z_k)`, strategy: 'tp', activationProduced: layer });
                activationProducedLayer = null; // Combined activation produced by COMM step
                break;

            case 'FFN':
                // 1. Linear 1 (Column Parallel)
                steps.push({ step: currentStep, type: 'COMPUTE', direction: 'forward', parallel: true, layer: layer, subStep: 'FFN1 (Col)', description: `TP Fwd: Compute sharded FFN Act H_k from ${prevLayerAct}.`, notation: `H_k = \\text{ColParallel}(${prevLayerAct}, W_{FFN1})`, strategy: 'tp' });
                // 2. Activation Fn (Local) - Often fused, we'll visually omit
                // 3. Linear 2 (Row Parallel)
                currentStep++;
                steps.push({ step: currentStep, type: 'COMPUTE', direction: 'forward', parallel: true, layer: layer, subStep: 'FFN2 (Row)', description: `TP Fwd: Compute partial Output Z_k from H_k.`, notation: `Z_k = \\text{RowParallel}(H_k, W_{FFN2})`, strategy: 'tp' });
                // 4. AllReduce Output
                currentStep++;
                steps.push({ step: currentStep, type: 'COMM', parallel: true, layer: layer, operation: 'AllReduce' as CommOperation, dataType: 'Activations' as CommDataType, subStep: 'FFN2 Reduce', description: `TP Fwd: AllReduce partial outputs Z_k.`, notation: `A_{${layer}} = \\text{AllReduce}_k(Z_k)`, strategy: 'tp', activationProduced: layer });
                 activationProducedLayer = null; // Combined activation produced by COMM step
                break;

            default: // Embed, LN, Output - Assume run replicated on both TP ranks for now (simplification)
                steps.push({ step: currentStep, type: 'COMPUTE', direction: 'forward', parallel: true, layer: layer, description: `TP Fwd: Processing ${layer} (Replicated).`, notation: `A_{${layer}} = \\text{Layer}(${prevLayerAct}, w_{${layer}})`, strategy: 'tp', activationProduced: layer });
                // activationProducedLayer remains set
                break;
        }
         // Store the layer name that produced the activation *at the end* of this sequence of steps for the layer
        // If activationProducedLayer is null (e.g., due to COMM), it means the COMM step was the producer.
        if(activationProducedLayer) {
             const lastStepForLayer = steps[steps.length - 1];
             if(lastStepForLayer && !lastStepForLayer.activationProduced) {
                 // Only set if not already set (e.g., by the COMM step)
                lastStepForLayer.activationProduced = activationProducedLayer;
             }
        }
    });

    // --- Conceptual Backward Pass ---
    // Add placeholder backward steps for activation tracking
    [...MODEL_LAYERS].reverse().forEach((layer) => {
         currentStep++;
         // Determine which activation is needed based on the model structure
         const fwdLayerIndex = MODEL_LAYERS.indexOf(layer);
         // Activation consumed is the one produced by the *previous* layer in the forward pass
         const actConsumedLayerName = fwdLayerIndex > 0 ? MODEL_LAYERS[fwdLayerIndex - 1] : 'Input'; // Input is consumed by first layer (Embed)

         // Backward pass for TP also involves AllReduce for certain gradient calculations (e.g., dX in ColParallel) - OMITTED FOR SIMPLICITY IN THIS CHUNK
          let description = `TP Bwd: Compute Grads for ${layer} (Placeholder).`;
          let notation = `\\nabla w_{${layer},k}, \\dots`;
          // Add simple differentiation for comm steps needed in backward (conceptual)
          if (layer === 'MHA' || layer === 'FFN') {
              // Backward of RowParallel needs AllReduce on input gradient
              // Backward of ColParallel needs AllReduce on weight gradient
               description = `TP Bwd: Compute Grads for ${layer} (Comm Placeholder).`;
               notation = `\\text{AllReduce}(\\dots)`
          }

         steps.push({ step: currentStep, type: 'COMPUTE', direction: 'backward', parallel: true, layer: layer, description: description, notation: notation, strategy: 'tp', activationConsumedLayer: actConsumedLayerName });
    });

    // --- Conceptual Update ---
    currentStep++;
    steps.push({ step: currentStep, type: 'UPDATE', parallel: true, layer: 'Optimizer', description: `TP Optimizer: Update sharded params $w^{(k)}$ (Placeholder).`, notation: `w^{(k)} \\leftarrow \\text{Optim}(w^{(k)}, \\nabla w^{(k)})`, strategy: 'tp' });
    currentStep++;
    steps.push({ step: currentStep, type: 'DONE', description: `TP Step Complete.`, strategy: 'tp' });

    // Re-assign step numbers sequentially after collecting all steps
    return steps.map((s, index) => ({ ...s, step: index }));
};
