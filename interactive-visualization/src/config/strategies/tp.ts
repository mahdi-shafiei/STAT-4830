import type { StepDetail, CommOperation, CommDataType, TensorInfo } from '../../context/types';

// Re-declare or import MODEL_LAYERS if not globally available
const MODEL_LAYERS = ['Embed', 'MHA', 'FFN', 'LN', 'Output']; // Consistent layer list
const N_tp = 2; // Fixed TP size for this chunk

export const generateTpSteps = (): StepDetail[] => {
    const steps: StepDetail[] = [];
    const Ntp = N_tp;
    // Define common tensor shapes (can be more dynamic later)
    const b = 'B', s = 'S', h = 'H', dff = '4H'; // Use symbolic dimensions
    const h_tp = `${h}/${Ntp}`;
    const dff_tp = `${dff}/${Ntp}`;

    steps.push({ step: 0, type: 'INIT', description: `TP (N=${Ntp}) Init: Layer params sharded.`, notation: `w_{L} = \\{ w_{L}^{(k)} \\}_{k=0}^{${Ntp-1}}`, strategy: 'tp', phase: 'idle' });

    let currentStep = 0;
    let previousActivation: TensorInfo = { label: '\\text{Input}', isSharded: false, numShards: Ntp, rows: b, cols: h }; // Assume input is replicated

    MODEL_LAYERS.forEach((layer) => {
        currentStep++; // Increment step counter for each layer's sequence
        let currentLayerInput = previousActivation;
        let activationProducedThisLayer: string | null = layer;
        let layerOutputTensor: TensorInfo = { label: `A_{${layer}}`, isSharded: false, numShards: Ntp, rows: b, cols: h }; // Default output shape

        switch (layer) {
            case 'MHA': // Simpler MHA viz for now, focus on Linear Ops
                 const qkvWeight: TensorInfo = { label: `W_{QKV}`, isSharded: 'col', numShards: Ntp, rows: h, cols: h };
                 const qkvIntermediate: TensorInfo = { label: `Q_k,K_k,V_k`, isSharded: 'col', numShards: Ntp, rows: b, cols: h_tp };
                 steps.push({ step: steps.length, type: 'COMPUTE', direction: 'forward', parallel: true, layer: layer, subStep: 'QKV (Col)', tpExecutionType: 'ColumnParallel', phase: 'compute', description: `Compute sharded Q_k, K_k, V_k`, notation: `Q_k, K_k, V_k = A_{prev} W_{QKV,k}`, inputTensor: currentLayerInput, weightTensor: qkvWeight, intermediateTensor: qkvIntermediate, strategy: 'tp' });

                 const attnInput = qkvIntermediate; // Use intermediate as input
                 const attnIntermediate: TensorInfo = { label: `O_k`, isSharded: 'col', numShards: Ntp, rows: b, cols: h_tp };
                 steps.push({ step: steps.length, type: 'COMPUTE', direction: 'forward', parallel: true, layer: layer, subStep: 'Attn Comp', tpExecutionType: 'LocalAttention', phase: 'compute', description: `Compute local Attention Output O_k.`, notation: `O_k = \\text{Attn}(Q_k,K_k,V_k)`, inputTensor: attnInput, weightTensor: {label: 'Attn Fn', isSharded: false, numShards: 1}, intermediateTensor: attnIntermediate, strategy: 'tp' });

                 const outProjInput = attnIntermediate;
                 const outProjWeight: TensorInfo = { label: `W_{Out}`, isSharded: 'row', numShards: Ntp, rows: h_tp, cols: h };
                 const outProjIntermediate: TensorInfo = { label: `Z_k`, isSharded: 'col', numShards: Ntp, rows: b, cols: h }; // Output of row matmul is sharded along output dim conceptually before sum
                 layerOutputTensor = { label: `A_{${layer}}`, isSharded: false, numShards: Ntp, rows: b, cols: h }; // Final output is replicated
                 steps.push({ step: steps.length, type: 'COMPUTE', direction: 'forward', parallel: true, layer: layer, subStep: 'Out (Row)', tpExecutionType: 'RowParallel', phase: 'compute', description: `Compute partial Output Projection Z_k.`, notation: `Z_k = O_k W_{Out,k}`, inputTensor: outProjInput, weightTensor: outProjWeight, intermediateTensor: outProjIntermediate, strategy: 'tp' });
                 steps.push({ step: steps.length, type: 'COMM', parallel: true, layer: layer, operation: 'AllReduce' as CommOperation, dataType: 'Activations' as CommDataType, subStep: 'Out Reduce', tpExecutionType: 'RowParallel', phase: 'comm_output', description: `AllReduce partial outputs Z_k.`, notation: `A_{${layer}} = \\text{AllReduce}_k(Z_k)`, inputTensor: outProjIntermediate, outputTensor: layerOutputTensor, strategy: 'tp', activationProduced: layer });
                 activationProducedThisLayer = null;
                 break;

             case 'FFN':
                const ffn1Weight: TensorInfo = { label: `W_{FFN1}`, isSharded: 'col', numShards: Ntp, rows: h, cols: dff };
                const ffn1Intermediate: TensorInfo = { label: `H_k`, isSharded: 'col', numShards: Ntp, rows: b, cols: dff_tp };
                const ffn2Weight: TensorInfo = { label: `W_{FFN2}`, isSharded: 'row', numShards: Ntp, rows: dff_tp, cols: h };
                const ffn2Intermediate: TensorInfo = { label: `Z_k`, isSharded: 'col', numShards: Ntp, rows: b, cols: h }; // Output before reduce
                layerOutputTensor = { label: `A_{${layer}}`, isSharded: false, numShards: Ntp, rows: b, cols: h }; // Final output

                // FFN1 (Column Parallel)
                // Phase: Comm Input (Broadcast - Implicit in TP Matmul)
                // Phase: Compute
                steps.push({ step: steps.length, type: 'COMPUTE', direction: 'forward', parallel: true, layer: layer, subStep: 'FFN1 (Col)', tpExecutionType: 'ColumnParallel', phase: 'compute', description: `Compute sharded FFN Act H_k.`, notation: `H_k = A_{prev} W_{FFN1,k}`, inputTensor: currentLayerInput, weightTensor: ffn1Weight, intermediateTensor: ffn1Intermediate, strategy: 'tp' });
                const ffn1OutputAsInput = ffn1Intermediate; // Use intermediate as input for next

                // FFN2 (Row Parallel)
                // Phase: Comm Input (Identity/Local - Input H_k is already sharded correctly)
                // Phase: Compute
                steps.push({ step: steps.length, type: 'COMPUTE', direction: 'forward', parallel: true, layer: layer, subStep: 'FFN2 (Row)', tpExecutionType: 'RowParallel', phase: 'compute', description: `Compute partial Output Z_k.`, notation: `Z_k = H_k W_{FFN2,k}`, inputTensor: ffn1OutputAsInput, weightTensor: ffn2Weight, intermediateTensor: ffn2Intermediate, strategy: 'tp' });
                // Phase: Comm Output (AllReduce)
                steps.push({ step: steps.length, type: 'COMM', parallel: true, layer: layer, operation: 'AllReduce' as CommOperation, dataType: 'Activations' as CommDataType, subStep: 'FFN2 Reduce', tpExecutionType: 'RowParallel', phase: 'comm_output', description: `AllReduce partial outputs Z_k.`, notation: `A_{${layer}} = \\text{AllReduce}_k(Z_k)`, inputTensor: ffn2Intermediate, outputTensor: layerOutputTensor, strategy: 'tp', activationProduced: layer });
                activationProducedThisLayer = null;
                break;

            default: // Embed, LN, Output - Replicated
                layerOutputTensor = { label: `A_{${layer}}`, isSharded: false, numShards: Ntp, rows: b, cols: h }; // Output is replicated
                steps.push({ step: steps.length, type: 'COMPUTE', direction: 'forward', parallel: true, layer: layer, tpExecutionType: 'Replicated', phase: 'compute', description: `Processing ${layer} (Replicated).`, notation: `${layerOutputTensor.label} = \\text{${layer}}(A_{prev}, w_{${layer}})`, inputTensor: currentLayerInput, weightTensor: {label: `w_{${layer}}`, isSharded: false, numShards: 1}, outputTensor: layerOutputTensor, strategy: 'tp', activationProduced: layer });
                break;
        }
         if(activationProducedThisLayer && steps.length > 0) { steps[steps.length - 1].activationProduced = activationProducedThisLayer; }
         previousActivation = layerOutputTensor; // Update for next layer's input
    });

    // --- Conceptual Backward Pass ---
     [...MODEL_LAYERS].reverse().forEach((layer) => {
         const fwdLayerIndex = MODEL_LAYERS.indexOf(layer);
         const actConsumedLayerName = fwdLayerIndex > 0 ? MODEL_LAYERS[fwdLayerIndex - 1] : null;
         const gradNotation = `\\nabla w_{${layer},k}, \\dots`; // Keep simple for now
         steps.push({ step: steps.length, type: 'COMPUTE', direction: 'backward', parallel: true, layer: layer, phase: 'compute', description: `TP Bwd: Compute Grads for ${layer}.`, notation: gradNotation, strategy: 'tp', activationConsumedLayer: actConsumedLayerName });
     });

    // --- Conceptual Update ---
    steps.push({ step: steps.length, type: 'UPDATE', parallel: true, layer: 'Optimizer', phase: 'compute', description: `TP Optimizer: Update params $w^{(k)}$.`, notation: `w^{(k)} \\leftarrow \\text{Optim}(w^{(k)}, \\nabla w_{Layer,k}^{(k)})`, strategy: 'tp' });
    steps.push({ step: steps.length, type: 'DONE', description: `TP Step Complete.`, strategy: 'tp', phase: 'idle' });

    return steps.map((s, index) => ({ ...s, step: index }));
};
