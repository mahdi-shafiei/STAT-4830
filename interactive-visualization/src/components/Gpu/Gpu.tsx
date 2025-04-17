import React from 'react';
import styles from './Gpu.module.css';
import MemoryBar from '../MemoryBar/MemoryBar';
import { GpuState, StepDetail } from '../../context/types';
import { motion } from 'framer-motion';

interface GpuProps extends GpuState {
    numGpusInGroup: number; // Total GPUs for calculating shard denom
    isParamsSharded: boolean;
    isGradsSharded: boolean;
    isOptStatesSharded: boolean;
    currentStepDetails: StepDetail | null;
}

const Gpu: React.FC<GpuProps> = ({
    id,
    paramMemory,
    activationMemory,
    gradientMemory,
    optStateMemory,
    status,
    currentLayerName,
    isParamsTempFull,
    dataShardId,
    numGpusInGroup,
    isParamsSharded,
    isGradsSharded,
    isOptStatesSharded,
    currentStepDetails
}) => {

    const shardDenom = numGpusInGroup > 0 ? numGpusInGroup : 1;
    const isTP = currentStepDetails?.strategy === 'tp';
    const tpShardDenom = isTP ? 2 : shardDenom; // TP fixed at 2 for now

    // Define max values
    const MAX_PARAM = 100;
    const MAX_ACTIVATION = 100;
    const MAX_GRADIENT = 100;
    const MAX_OPTSTATE = 100;

    // --- Enhanced Status Text Logic ---
    let statusText = 'Idle';
    const layerFromState = currentLayerName; // Keep original state name as fallback
    const layerName = currentStepDetails?.layer || layerFromState; // Prefer details for current step

    if (status === 'computing' && layerName) {
        let stepDesc = '';
        if (isTP && currentStepDetails?.tpExecutionType) {
            stepDesc = `(${currentStepDetails.tpExecutionType})`; // Use TP type if present
        } else if (currentStepDetails?.subStep) {
            stepDesc = `(${currentStepDetails.subStep})`;
        }
        statusText = `Compute: ${layerName} ${stepDesc}`;

        // Add notation if available and it's a compute step
        if (currentStepDetails?.type === 'COMPUTE' && currentStepDetails.notation) {
            // Keep it concise for the status box
             const conciseNotation = currentStepDetails.notation.split('=')[0]; // Just show LHS for brevity
             statusText += ` [${conciseNotation?.trim() ?? ''}]`;
        }

    } else if (status === 'communicating' && currentStepDetails?.operation) {
        statusText = `${currentStepDetails.operation} (${currentStepDetails.dataType || ''})...`;
        // Add notation for communication if useful
        if (currentStepDetails.notation) {
             statusText += ` [${currentStepDetails.notation}]`;
        }
    } else if (status === 'communicating') {
        statusText = 'Communicating...';
    } else {
        statusText = status.charAt(0).toUpperCase() + status.slice(1); // Default: Capitalize status
    }

    return (
        <motion.div
            className={styles.gpuContainer}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: id * 0.05 }}
            layout
        >
            <div className={styles.gpuHeader}>
                 GPU {id} {dataShardId && !isTP ? `(Data Shard ${dataShardId})` : ''} {isTP ? `(TP Rank ${id})` : ''}
            </div>
            <div className={styles.memorySection}>
                <MemoryBar type="Params" value={paramMemory} gpuId={id} isSharded={isParamsSharded || isTP} shardDenom={isTP ? tpShardDenom : shardDenom} isTempFull={isParamsTempFull} maxValue={MAX_PARAM} />
                <MemoryBar type="Activations" value={activationMemory} gpuId={id} isSharded={false} shardDenom={1} maxValue={MAX_ACTIVATION} />
                <MemoryBar type="Gradients" value={gradientMemory} gpuId={id} isSharded={isGradsSharded || isTP} shardDenom={isTP ? tpShardDenom : shardDenom} maxValue={MAX_GRADIENT} />
                <MemoryBar type="OptStates" value={optStateMemory} gpuId={id} isSharded={isOptStatesSharded || isTP} shardDenom={isTP ? tpShardDenom : shardDenom} maxValue={MAX_OPTSTATE} />
            </div>
            <div className={`${styles.statusIndicator} ${styles[status]}`} title={currentStepDetails?.notation || statusText}>
                {statusText}
            </div>
        </motion.div>
    );
};

export default Gpu;
