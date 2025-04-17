import React from 'react';
import Gpu from '../Gpu/Gpu';
import CommunicationArrow from '../CommunicationArrow/CommunicationArrow';
import NotationDisplay from '../MathDisplay/MathDisplay';
import styles from './VisualizationArea.module.css';
import { useSimulation } from '../../context/SimulationContext';
import { AnimatePresence } from 'framer-motion';
import TpLayerExecutionViz from '../TpLayerExecutionViz/TpLayerExecutionViz';
import type { TpStepInfo, TpOperationType } from '../TpLayerExecutionViz/TpLayerExecutionViz';
import { useGpuPositions, Point } from '../../hooks/useGpuPositions';
import { useRef } from 'react';
import { DetailedTpLinearOpViz } from '../DetailedTpOperationViz/DetailedTpOperationViz';
import type { OperationPhase, TensorInfo } from '../DetailedTpOperationViz/DetailedTpOperationViz';
import { BroadcastAnim } from '../CommunicationAnimations/BroadcastAnim';
import { ScatterAnim } from '../CommunicationAnimations/ScatterAnim';
import { AllReduceAnim } from '../CommunicationAnimations/AllReduceAnim';
import { AllGatherAnim } from '../CommunicationAnimations/AllGatherAnim';

const VisualizationArea: React.FC = () => {
  const { gpuStates, stepDetails, numGpus, strategy, currentStep } = useSimulation();

  const gpuContainerRef = useRef<HTMLDivElement>(null);
  const { positions: gpuDomPositions, containerRect } = useGpuPositions(gpuContainerRef, numGpus, currentStep);

  // Determine if communication is happening based on stepDetails
  const isCommunicating = stepDetails?.type === 'COMM';
  const commDataType = stepDetails?.dataType;
  const commOperation = stepDetails?.operation;
  const commType = stepDetails?.operation; // Using commType for clarity in arrow logic

  // Calculate center position (needed for some animations)
  const calculatedCenterPos = containerRect ? {
       x: containerRect.width / 2,
       y: containerRect.height / 2 - 80 // Adjust Y offset to be above GPUs
  } : { x: 300, y: 150 }; // Fallback center

  // Determine TP operation/phase for Detailed Viz
  let detailedTpOperation: 'ColumnLinear' | 'RowLinear' | null = null;
  let detailedTpPhase: OperationPhase = 'idle';
  let inputTensor = stepDetails?.inputTensor;
  let weightTensor = stepDetails?.weightTensor;
  let outputTensor = stepDetails?.outputTensor;
  let intermediateTensor = stepDetails?.intermediateTensor;

  if (strategy === 'tp' && stepDetails?.tpExecutionType) {
      detailedTpPhase = stepDetails.phase || 'compute'; // Get phase from step detail
      const execType = stepDetails.tpExecutionType;
      if (execType === 'ColumnParallel') {
          detailedTpOperation = 'ColumnLinear';
           // Column Linear doesn't have explicit comm steps in our simplified tp.ts
           // but we can show compute phase
           if(detailedTpPhase === 'compute') {
               // input/weight/intermediate are set by step generator
           } else { detailedTpOperation = null; } // Hide viz if not compute phase

      } else if (execType === 'RowParallel') {
          detailedTpOperation = 'RowLinear';
           // Viz handles display based on compute/comm_output phase prop
           if(detailedTpPhase === 'compute') {
                // input/weight/intermediate set by step generator
           } else if (detailedTpPhase === 'comm_output') {
                // input is intermediate, output is final. weight maybe hidden.
                inputTensor = stepDetails.inputTensor; // Should be Z_k from COMM step
                outputTensor = stepDetails.outputTensor; // Should be final A_layer
                weightTensor = undefined; // Hide weight during communication viz
                intermediateTensor = undefined;
           } else { detailedTpOperation = null; } // Hide if not compute/comm_output

      } else {
           detailedTpOperation = null; // Don't show for LocalAttention/Replicated yet
      }
  }

  return (
    <div className={styles.visualizationArea}>
        {/* Layer Icons (Top) - If you add them, they would go here */}
        {/* Placeholder for layer icons area */}

        {/* Render NEW Detailed TP Linear Op Viz */}
        <DetailedTpLinearOpViz
            isActive={detailedTpOperation !== null}
            operation={detailedTpOperation}
            phase={detailedTpPhase}
            Ntp={numGpus}
            inputTensor={inputTensor}
            weightTensor={weightTensor}
            outputTensor={outputTensor}
            intermediateTensor={intermediateTensor}
        />

        {/* --- GPU Container --- */}
        <div ref={gpuContainerRef} className={styles.gpuContainer} style={{ gridTemplateColumns: `repeat(${Math.min(numGpus, 4)}, 1fr)` }}>
             <AnimatePresence>
                {gpuStates.map((gpuState, index) => {
                    // Determine sharding flags based on strategy
                    const isParamsSharded = strategy === 'fsdp' || strategy === 'tp';
                    // Adjust grad sharding - TP grads are conceptually sharded, display handled in MemoryBar/Gpu
                    const isGradsSharded = strategy === 'fsdp' || strategy === 'tp';
                    const isOptStatesSharded = strategy === 'fsdp' || strategy === 'tp';

                    return (
                        <Gpu
                            key={gpuState.id}
                            {...gpuState} // Spread existing state
                            numGpusInGroup={numGpus} // Pass total GPU count for context
                            isParamsSharded={isParamsSharded}
                            isGradsSharded={isGradsSharded} // Pass updated grad sharding flag
                            isOptStatesSharded={isOptStatesSharded}
                            currentStepDetails={stepDetails} // Pass current step details
                            strategy={strategy} // Pass strategy down
                        />
                    );
                })}
            </AnimatePresence>
        </div>

        {/* NEW: Render Specific Communication Animations */}
        {/* These rely on accurate gpuDomPositions */} 
        {strategy === 'tp' && commOperation === 'Broadcast' && (
             <BroadcastAnim isActive={isCommunicating} sourcePos={calculatedCenterPos} targetPositions={gpuDomPositions.map(p => p.center)} dataType={commDataType}/>
        )}
        {strategy === 'tp' && commOperation === 'Scatter' && (
             <ScatterAnim isActive={isCommunicating} sourcePos={calculatedCenterPos} targetPositions={gpuDomPositions.map(p => p.center)} dataType={commDataType}/>
        )}
         {strategy === 'tp' && commOperation === 'AllReduce' && (
             <AllReduceAnim isActive={isCommunicating} gpuPositions={gpuDomPositions.map(p => p.center)} centerPos={calculatedCenterPos} dataType={commDataType}/>
         )}
         {strategy === 'tp' && commOperation === 'AllGather' && (
             <AllGatherAnim isActive={isCommunicating} gpuPositions={gpuDomPositions.map(p => p.center)} dataType={commDataType}/>
         )}

        {/* Keep non-TP communication arrow if needed */}
        <AnimatePresence>
         {isCommunicating && strategy !== 'tp' && commOperation && commDataType && (
              <CommunicationArrow key="comm-arrow" type={commOperation as any} dataType={commDataType as any} isActive={true} />
          )}
         </AnimatePresence>

        {/* Display KaTeX notation - Potentially integrate into OperationDetailsPanel later */}
        {/* <NotationDisplay notation={stepDetails?.notation || ''} description={stepDetails?.description || ''} /> */}
    </div>
  );
};

export default VisualizationArea;
