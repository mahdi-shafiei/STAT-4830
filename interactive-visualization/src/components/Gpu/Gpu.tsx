import React from 'react';
import styles from './Gpu.module.css';
import MemoryBar from '../MemoryBar/MemoryBar';
import MathDisplay from '../MathDisplay/MathDisplay';
import type { GpuState, StepDetail } from '../../context/SimulationContext';
import { motion } from 'framer-motion';

export interface GpuProps extends GpuState {
    numGpusInGroup: number;
    isParamsSharded: boolean;
    isGradsSharded: boolean;
    isOptStatesSharded: boolean;
    currentStepDetails?: StepDetail | null;
    strategy?: string;
    layerFromState?: string;
}

const MAX_MEMORY_PER_TYPE = { Params: 100, Activations: 100, Gradients: 100, OptStates: 100 };

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
    const isProcessing = status === 'computing';
    const isCommunicating = status === 'communicating';
    const gpuClasses = [
        styles.gpu,
        isProcessing ? styles.processing : '',
        isCommunicating ? styles.communicating : ''
    ].filter(Boolean).join(' ');

    const shardDenom = numGpusInGroup > 0 ? numGpusInGroup : 1;
    const tpShardDenom = strategy === 'tp' ? 2 : shardDenom;

    let statusPrefix = 'Idle';
    let statusNotation = currentStepDetails?.notation || '';
    const layerName = currentStepDetails?.layer || layerFromState;

    if (status === 'computing' && layerName) {
        let stepDesc = '';
        if (currentStepDetails?.tpExecutionType) {
            stepDesc = `(${currentStepDetails.tpExecutionType})`;
        } else if (currentStepDetails?.subStep) {
            stepDesc = `(${currentStepDetails.subStep})`;
        }
        const batchShardText = dataShardId !== undefined ? ` B_{${dataShardId}}` : '';
        statusPrefix = `Compute${batchShardText}: ${layerName} ${stepDesc}`;
    } else if (status === 'communicating' && currentStepDetails?.operation) {
        statusPrefix = `${currentStepDetails.operation} (${currentStepDetails.dataType || 'data'})...`;
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
            data-gpu-id={gpuId}
        >
            <div className={styles.gpuHeader}>
                GPU {gpuId} {dataShardId && strategy !== 'tp' ? `(Data Shard ${dataShardId})` : ''} {strategy === 'tp' ? `(TP Rank ${gpuId})` : ''} {status !== 'idle' ? '(Active)' : ''}
            </div>
            <div className={styles.memorySection}>
                <MemoryBar type="Params" value={paramMemory} maxValue={MAX_MEMORY_PER_TYPE.Params} shardDenom={numGpusInGroup} isSharded={isParamsSharded} isTempFull={isParamsTempFull} gpuId={gpuId} />
                <MemoryBar type="Activations" value={activationMemory} maxValue={MAX_MEMORY_PER_TYPE.Activations} shardDenom={1} isSharded={false} gpuId={gpuId} />
                <MemoryBar type="Gradients" value={gradientMemory} maxValue={MAX_MEMORY_PER_TYPE.Gradients} shardDenom={numGpusInGroup} isSharded={isGradsSharded} gpuId={gpuId} />
                <MemoryBar type="OptStates" value={optStateMemory} maxValue={MAX_MEMORY_PER_TYPE.OptStates} shardDenom={numGpusInGroup} isSharded={isOptStatesSharded} gpuId={gpuId} />
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
