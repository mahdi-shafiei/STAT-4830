import React from 'react';
import styles from './Gpu.module.css';
import MemoryBar from '../MemoryBar/MemoryBar';
import MathDisplay from '../MathDisplay/MathDisplay';
import type { GpuState, StepDetail } from '../../context/types';
import { motion } from 'framer-motion';

interface GpuProps extends GpuState {
    numGpusInGroup: number; // Total GPUs for calculating shard denom
    isParamsSharded: boolean;
    isGradsSharded: boolean;
    isOptStatesSharded: boolean;
    currentStepDetails: StepDetail | null;
    strategy?: string;
    layerFromState?: string;
}

const MAX_MEMORY_PER_TYPE = {
    param: 100,
    activation: 100,
    gradient: 100,
    optState: 100,
};

const Gpu: React.FC<GpuProps> = ({
    id: gpuId,
    paramMemory,
    activationMemory,
    gradientMemory,
    optStateMemory,
    status,
    currentLayerName: layerFromState,
    isParamsTempFull,
    dataShardId,
    numGpusInGroup,
    isParamsSharded,
    isGradsSharded,
    isOptStatesSharded,
    currentStepDetails,
    strategy
}) => {

    const gpuClasses = `${styles.gpuContainer} ${styles[status] || ''}`;

    const shardDenom = numGpusInGroup > 0 ? numGpusInGroup : 1;
    const tpShardDenom = strategy === 'tp' ? 2 : shardDenom;

    let statusPrefix = 'Idle';
    let statusNotation = '';
    const layerName = currentStepDetails?.layer || layerFromState;

    if (status === 'computing' && layerName) {
        let stepDesc = '';
        if (currentStepDetails?.tpExecutionType) {
            stepDesc = `(${currentStepDetails.tpExecutionType})`;
        } else if (currentStepDetails?.subStep) {
            stepDesc = `(${currentStepDetails.subStep})`;
        }
        statusPrefix = `Compute: ${layerName} ${stepDesc}`;
        if (currentStepDetails?.type === 'COMPUTE' && currentStepDetails.notation) {
            statusNotation = currentStepDetails.notation;
        }
    } else if (status === 'communicating' && currentStepDetails?.operation) {
        statusPrefix = `${currentStepDetails.operation} (${currentStepDetails.dataType || 'data'})...`;
        if (currentStepDetails.notation) {
            statusNotation = currentStepDetails.notation;
        }
    } else if (status === 'communicating') {
        statusPrefix = 'Communicating...';
    } else if (status !== 'idle') {
        statusPrefix = status.charAt(0).toUpperCase() + status.slice(1);
    }

    return (
        <motion.div
            className={gpuClasses}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: gpuId * 0.05 }}
            layout
        >
            <div className={styles.gpuHeader}>
                GPU {gpuId} {dataShardId && strategy !== 'tp' ? `(Data Shard ${dataShardId})` : ''} {strategy === 'tp' ? `(TP Rank ${gpuId})` : ''} {status !== 'idle' ? '(Active)' : ''}
            </div>
            <div className={styles.memorySection}>
                <MemoryBar type="Params" value={paramMemory} gpuId={gpuId} isSharded={isParamsSharded || strategy === 'tp'} shardDenom={strategy === 'tp' ? tpShardDenom : shardDenom} isTempFull={isParamsTempFull} maxValue={MAX_MEMORY_PER_TYPE.param} />
                <MemoryBar type="Activations" value={activationMemory} gpuId={gpuId} isSharded={false} shardDenom={1} maxValue={MAX_MEMORY_PER_TYPE.activation} />
                <MemoryBar type="Gradients" value={gradientMemory} gpuId={gpuId} isSharded={isGradsSharded || strategy === 'tp'} shardDenom={strategy === 'tp' ? tpShardDenom : shardDenom} maxValue={MAX_MEMORY_PER_TYPE.gradient} />
                <MemoryBar type="OptStates" value={optStateMemory} gpuId={gpuId} isSharded={isOptStatesSharded || strategy === 'tp'} shardDenom={strategy === 'tp' ? tpShardDenom : shardDenom} maxValue={MAX_MEMORY_PER_TYPE.optState} />
            </div>
            <div className={styles.computeSection}>
                <div className={styles.statusText}>{statusPrefix}</div>
                {statusNotation && (
                    <div className={styles.statusNotation} title={statusNotation}>
                        <MathDisplay texString={statusNotation} />
                    </div>
                )}
            </div>
        </motion.div>
    );
};

export default Gpu;
