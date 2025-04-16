import React from 'react';
import styles from './Gpu.module.css';
import MemoryBar from '../MemoryBar/MemoryBar';

// Define expected memory types
type MemoryType = 'Params' | 'Activations' | 'Gradients' | 'OptStates';

export interface GpuProps {
  gpuId: number;
  paramMemory: number;
  activationMemory: number;
  gradientMemory: number; // Ensure this is passed
  optStateMemory: number;
  isProcessing?: boolean;
  isCommunicating?: boolean; // Flag for communication state
  currentLayerName?: string;
}

// Max values for memory bars (can be adjusted)
const MAX_MEMORY_PER_TYPE: Record<MemoryType, number> = {
    Params: 100,
    Activations: 100,
    Gradients: 100, // Max for gradients bar
    OptStates: 100
};

const Gpu: React.FC<GpuProps> = ({
  gpuId,
  paramMemory,
  activationMemory,
  gradientMemory,
  optStateMemory,
  isProcessing,
  isCommunicating, // Use the flag
  currentLayerName
}) => {
  // Determine combined class name for styling based on state
  const gpuClasses = [
      styles.gpu,
      isProcessing ? styles.processing : '',
      isCommunicating ? styles.communicating : '' // Add class for communication visual state
  ].filter(Boolean).join(' '); // Filter out empty strings and join

  // Determine the status text to display
  let statusText = 'Idle';
  if (isProcessing && currentLayerName) {
      statusText = `Computing: ${currentLayerName}`;
  } else if (isCommunicating) {
      statusText = 'Communicating...';
  }

  return (
    <div className={gpuClasses}>
      <div className={styles.gpuHeader}>GPU {gpuId} {isProcessing || isCommunicating ? '(Active)' : ''}</div>
      <div className={styles.memorySection}>
        <MemoryBar type="Params" value={paramMemory} maxValue={MAX_MEMORY_PER_TYPE.Params} />
        <MemoryBar type="Activations" value={activationMemory} maxValue={MAX_MEMORY_PER_TYPE.Activations} />
        {/* Render Gradient Memory Bar */}
        <MemoryBar type="Gradients" value={gradientMemory} maxValue={MAX_MEMORY_PER_TYPE.Gradients} />
        <MemoryBar type="OptStates" value={optStateMemory} maxValue={MAX_MEMORY_PER_TYPE.OptStates} />
      </div>
      <div className={styles.computeSection}>
         {statusText} {/* Display dynamic status */}
      </div>
    </div>
  );
};

export default Gpu;
