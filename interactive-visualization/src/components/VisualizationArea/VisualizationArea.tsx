import React from 'react';
import Gpu from '../Gpu/Gpu';
import ModelLayer from '../ModelLayer/ModelLayer';
// import DataPacket from '../DataPacket/DataPacket'; // Packet less relevant for DP overview
import CommunicationArrow from '../CommunicationArrow/CommunicationArrow'; // Import
import styles from './VisualizationArea.module.css';
import { useSimulation } from '../../context/SimulationContext'; // Import hook and types if exported
import type { GpuState, CommOperation, CommDataType } from '../../context/SimulationContext';
import { AnimatePresence } from 'framer-motion'; // For communication animation

// Layers definition can be moved to context or config later
const MODEL_LAYERS = ['Embed', 'MHA', 'FFN', 'LN', 'Output'];

const VisualizationArea: React.FC = () => {
  // Get comprehensive state from context
  const {
      numGpus, // Actual number of GPUs for the simulation
      gpuStates, // Array of states for each GPU
      stepDetails,
     } = useSimulation();

  // Determine communication state based on current step details
  const isCommunicating = stepDetails?.type === 'COMM';
  const commType = isCommunicating ? stepDetails.operation as CommOperation : undefined;
  const commDataType = isCommunicating ? stepDetails.dataType as CommDataType : undefined;

  // Find the current layer being processed (if any GPU is computing)
  const currentProcessingLayer = gpuStates?.find(gpu => gpu.status === 'computing')?.currentLayerName;

  return (
    <div className={styles.visualizationArea}>
      {/* Optional: Layer Visualization - less critical for DP focus */}
      <div className={styles.layerContainerWrapper}>
          <div className={styles.layerContainer}>
            {MODEL_LAYERS.map((layerType) => (
              <ModelLayer
                 key={layerType}
                 type={layerType}
                 // Highlight if this layer is currently being processed by any GPU
                 isHighlighted={currentProcessingLayer === layerType}
                 />
            ))}
          </div>
      </div>

      {/* GPU Container - Renders based on context's gpuStates */}
      {/* Use CSS grid properties for layout */}
      <div className={styles.gpuContainer} style={{ gridTemplateColumns: `repeat(${Math.min(numGpus, 4)}, minmax(180px, 1fr))`, }}> {/* Max 4 per row */}
        {/* Ensure gpuStates is an array before mapping */}
        {Array.isArray(gpuStates) && gpuStates.map((gpuState: GpuState) => (
          <Gpu
            key={gpuState.id}
            gpuId={gpuState.id}
            paramMemory={gpuState.paramMemory}
            activationMemory={gpuState.activationMemory}
            gradientMemory={gpuState.gradientMemory}
            optStateMemory={gpuState.optStateMemory}
            isProcessing={gpuState.status === 'computing'}
            isCommunicating={gpuState.status === 'communicating'}
            currentLayerName={gpuState.currentLayerName}
         />
        ))}
      </div>

       {/* Communication Visualization - Overlay */}
       <AnimatePresence>
        {isCommunicating && commType && commDataType && (
            <CommunicationArrow
                key="comm-arrow" // Key for AnimatePresence
                type={commType}
                dataType={commDataType}
                isActive={true}
            />
        )}
       </AnimatePresence>
    </div>
  );
};

export default VisualizationArea;
