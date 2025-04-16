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

    // Define max values (could be passed as props or from context if dynamic)
    const MAX_PARAM = 100;
    const MAX_ACTIVATION = 100;
    const MAX_GRADIENT = 100;
    const MAX_OPTSTATE = 100;

    return (
        <motion.div
            className={styles.gpuContainer}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: id * 0.05 }}
            layout
        >
            <div className={styles.gpuHeader}>
                GPU {id} {dataShardId ? `(Data Shard ${dataShardId})` : ''}
            </div>
            <div className={styles.memorySection}>
                <MemoryBar type="Params" value={paramMemory} gpuId={id} isSharded={isParamsSharded} shardDenom={isTP ? tpShardDenom : shardDenom} isTempFull={isParamsTempFull} maxValue={MAX_PARAM} />
                <MemoryBar type="Activations" value={activationMemory} gpuId={id} isSharded={false} shardDenom={1} maxValue={MAX_ACTIVATION} />{/* Activations not visually sharded yet */}
                <MemoryBar type="Gradients" value={gradientMemory} gpuId={id} isSharded={isGradsSharded || isTP} shardDenom={isTP ? tpShardDenom : shardDenom} maxValue={MAX_GRADIENT} />
                <MemoryBar type="OptStates" value={optStateMemory} gpuId={id} isSharded={isOptStatesSharded || isTP} shardDenom={isTP ? tpShardDenom : shardDenom} maxValue={MAX_OPTSTATE} />
            </div>
            <div className={`${styles.statusIndicator} ${styles[status]}`}>
                {status.toUpperCase()}: {currentLayerName || 'N/A'}
            </div>
        </motion.div>
    );
};

export default Gpu;
