import React from 'react';
import styles from './Gpu.module.css';
import MemoryBar from '../MemoryBar/MemoryBar';
import type { GpuState, StepDetail } from '../../context/SimulationContext';

export interface GpuProps extends GpuState {
  numGpusInGroup: number;
  isParamsSharded: boolean;
  isGradsSharded: boolean;
  isOptStatesSharded: boolean;
  currentStepDetails?: StepDetail | null;
}

const MAX_MEMORY_PER_TYPE = { Params: 100, Activations: 100, Gradients: 100, OptStates: 100 };

const Gpu: React.FC<GpuProps> = ({
  id: gpuId, // Use id prop as gpuId
  paramMemory, activationMemory, gradientMemory, optStateMemory,
  status, currentLayerName: layerFromState, isParamsTempFull, dataShardId,
  numGpusInGroup,
  isParamsSharded, isGradsSharded, isOptStatesSharded,
  currentStepDetails
}) => {
  const isProcessing = status === 'computing';
  const isCommunicating = status === 'communicating';
  const gpuClasses = [ styles.gpu, isProcessing ? styles.processing : '', isCommunicating ? styles.communicating : '' ].filter(Boolean).join(' ');
  let statusText = 'Idle';
  const currentLayer = currentStepDetails?.layer || layerFromState;

  // Determine status text (using B_k notation for shards)
  if (status === 'computing' && currentLayer) {
      const batchShardText = dataShardId !== undefined ? `B${dataShardId}` : '';
      if (currentStepDetails?.direction === 'forward' && batchShardText) statusText = `Compute ${batchShardText}: Fwd-${currentLayer}`;
      else if (currentStepDetails?.direction === 'backward') statusText = `Compute Bwd: ${currentLayer}`;
      else if (currentLayer === 'Optimizer') statusText = `Computing: ${currentLayer}`;
      else if (currentLayer === 'Gradients') statusText = `Compute Grads g_${gpuId}`; // Use g_k notation
      else statusText = `Computing: ${currentLayer}`;
  } else if (status === 'communicating' && currentStepDetails?.operation) { statusText = `${currentStepDetails.operation} (${currentStepDetails.dataType})...`;
  } else if (status === 'communicating') { statusText = 'Communicating...'; }

  return (
    <div className={gpuClasses}>
      <div className={styles.gpuHeader}>GPU {gpuId} {status !== 'idle' ? '(Active)' : ''}</div>
      <div className={styles.memorySection}>
        {/* FIX: Pass gpuId to MemoryBar */}
        <MemoryBar type="Params" value={paramMemory} maxValue={MAX_MEMORY_PER_TYPE.Params} shardDenom={numGpusInGroup} isSharded={isParamsSharded} isTempFull={isParamsTempFull} gpuId={gpuId} />
        <MemoryBar type="Activations" value={activationMemory} maxValue={MAX_MEMORY_PER_TYPE.Activations} shardDenom={1} isSharded={false} gpuId={gpuId} />
        <MemoryBar type="Gradients" value={gradientMemory} maxValue={MAX_MEMORY_PER_TYPE.Gradients} shardDenom={numGpusInGroup} isSharded={isGradsSharded} gpuId={gpuId} />
        <MemoryBar type="OptStates" value={optStateMemory} maxValue={MAX_MEMORY_PER_TYPE.OptStates} shardDenom={numGpusInGroup} isSharded={isOptStatesSharded} gpuId={gpuId} />
      </div>
      <div className={styles.computeSection}> {statusText} </div>
    </div>
  );
};
export default Gpu;
