// Assuming StepDetail is exported from SimulationContext.tsx
// If not, define it here or import appropriately
import type { StepDetail, CommOperation, CommDataType } from '../../context/SimulationContext';

const MODEL_LAYERS = ['Embed', 'MHA', 'FFN', 'LN', 'Output']; // Keep consistent

// Generates the sequence of steps for Data Parallelism
export const generateDpSteps = (): StepDetail[] => {
    const steps: StepDetail[] = [];

    // Step 0: Initial State
    steps.push({
        step: 0,
        type: 'INIT',
        description: 'DP Initial State: Model replicated. Ready.'
    });

    // Step 1 -> N: Parallel Forward Pass
    MODEL_LAYERS.forEach((layer, index) => {
        steps.push({
            step: index + 1,
            type: 'COMPUTE',
            direction: 'forward',
            parallel: true, // Indicate parallel execution across DP ranks
            layer: layer, // Layer being processed
            description: `Parallel Forward Pass: Processing ${layer} layer on all GPUs.`
        });
    });

    // Step N+1: Forward Pass Complete / Placeholder Gradient Computation
    const forwardEndStep = MODEL_LAYERS.length + 1;
    steps.push({
        step: forwardEndStep,
        type: 'GRADIENTS',
        parallel: true,
        description: `Forward Complete. Gradients computed locally on each GPU (Placeholder).`
    });

    // Step N+2: AllReduce Communication
    steps.push({
        step: forwardEndStep + 1,
        type: 'COMM',
        parallel: true, // Operation involves all GPUs
        operation: 'AllReduce' as CommOperation,
        dataType: 'Gradients' as CommDataType,
        description: `Communication: Averaging gradients across all GPUs via AllReduce.`
    });

    // Step N+3: Optimizer Update
    steps.push({
        step: forwardEndStep + 2,
        type: 'UPDATE',
        parallel: true,
        layer: 'Optimizer', // Conceptual layer name for update
        description: `Optimizer Step: Updating model replicas identically on all GPUs.`
    });

     // Step N+4: Final State
    steps.push({
        step: forwardEndStep + 3,
        type: 'DONE',
        description: `DP Step Complete. Ready for next iteration.`
    });


    // Re-assign step numbers sequentially starting from 0
    return steps.map((s, index) => ({ ...s, step: index }));
};

// Export the generated steps and total count
export const DP_STEPS = generateDpSteps();
// Calculate total steps based on the length (last valid index)
export const DP_TOTAL_STEPS = DP_STEPS.length > 0 ? DP_STEPS.length - 1 : 0;
