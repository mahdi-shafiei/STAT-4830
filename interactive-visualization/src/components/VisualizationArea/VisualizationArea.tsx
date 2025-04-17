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
import { useRef, useMemo } from 'react';
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

  // Get container ref offset once
  // NOTE: This offset calculation is relative to viewport, not the parent div.
  // For SVG overlay positioned absolute 0,0 within VisualizationArea, we likely don't need this.
  // Let's start with containerOffset = {x:0, y:0} and adjust if needed.
  // const containerOffset = useMemo(() => {
  //     if (gpuContainerRef.current) {
  //         const rect = gpuContainerRef.current.getBoundingClientRect();
  //         // Get the offset of the parent (.visualizationArea) to make it relative to the parent
  //         const parentRect = gpuContainerRef.current.parentElement?.getBoundingClientRect();
  //         const offsetX = rect.left - (parentRect?.left ?? 0);
  //         const offsetY = rect.top - (parentRect?.top ?? 0);
  //         return { x: offsetX, y: offsetY };
  //     }
  //     return { x: 0, y: 0 };
  // }, [gpuContainerRef.current]); // Depends only on the ref

  // Use calculated center relative to the *container* for CenterPos
  // Use containerRect from the hook if available
  const calculatedCenterPos = containerRect ? {
    x: containerRect.width / 2,
    y: containerRect.height / 2 - 50 // Position above GPU centers
  } : { x: 300, y: 150 }; // Fallback

  // Filter out null positions before passing
  // Ensure the Point type includes center
  const validGpuCenters: Point[] = gpuDomPositions.filter(p => p !== null).map(p => p.center);

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
           if(detailedTpPhase !== 'compute') { detailedTpOperation = null; }
      } else if (execType === 'RowParallel') {
          detailedTpOperation = 'RowLinear';
           if(detailedTpPhase === 'compute') {
                // Compute phase logic
           } else if (detailedTpPhase === 'comm_output') {
                inputTensor = stepDetails.inputTensor;
                outputTensor = stepDetails.outputTensor;
                weightTensor = undefined;
                intermediateTensor = undefined;
           } else { detailedTpOperation = null; }
      } else {
           detailedTpOperation = null;
      }
  }

  return (
    <div className={styles.visualizationArea}>
        {/* Layer Icons (Top) - If you add them, they would go here */}
        {/* Placeholder for layer icons area */}

        {/* Render NEW Detailed TP Linear Op Viz */}
        <AnimatePresence>
            {detailedTpOperation !== null && (
                <DetailedTpLinearOpViz
                    key={detailedTpOperation + detailedTpPhase}
                    isActive={true}
                    operation={detailedTpOperation}
                    phase={detailedTpPhase}
                    Ntp={numGpus}
                    inputTensor={inputTensor}
                    weightTensor={weightTensor}
                    outputTensor={outputTensor}
                    intermediateTensor={intermediateTensor}
                />
            )}
        </AnimatePresence>

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

        {/* Communication Animations Container (relative to VisualizationArea) */}
        {/* Pass containerOffset = {x:0, y:0} since the SVG is positioned relative to this container */}
        <div className={styles.communicationAnimationContainer}>
            {strategy === 'tp' && commOperation === 'Broadcast' && (
                 <BroadcastAnim isActive={isCommunicating} sourcePos={calculatedCenterPos} targetPositions={validGpuCenters} dataType={commDataType} containerOffset={{x:0, y:0}}/>
            )}
             {strategy === 'tp' && commOperation === 'Scatter' && (
                 <ScatterAnim isActive={isCommunicating} sourcePos={calculatedCenterPos} targetPositions={validGpuCenters} dataType={commDataType} containerOffset={{x:0, y:0}}/>
             )}
              {strategy === 'tp' && commOperation === 'AllReduce' && (
                  <AllReduceAnim isActive={isCommunicating} gpuPositions={validGpuCenters} centerPos={calculatedCenterPos} dataType={commDataType} containerOffset={{x:0, y:0}}/>
              )}
              {strategy === 'tp' && commOperation === 'AllGather' && (
                  <AllGatherAnim isActive={isCommunicating} gpuPositions={validGpuCenters} dataType={commDataType} containerOffset={{x:0, y:0}}/>
              )}

              {/* Add DP / FSDP Communication Animations Here Later */}
              {/* Example for DP AllReduce */}
               {strategy === 'dp' && commOperation === 'AllReduce' && (
                  <AllReduceAnim isActive={isCommunicating} gpuPositions={validGpuCenters} centerPos={calculatedCenterPos} dataType={commDataType} containerOffset={{x:0, y:0}}/>
               )}
              {/* Example for FSDP AllGather */}
               {strategy === 'fsdp' && commOperation === 'AllGather' && (
                  <AllGatherAnim isActive={isCommunicating} gpuPositions={validGpuCenters} dataType={commDataType} containerOffset={{x:0, y:0}}/>
               )}
               {/* Example for FSDP ReduceScatter */}
                {strategy === 'fsdp' && commOperation === 'ReduceScatter' && (
                   // Need a ReduceScatterAnim component
                   // <ReduceScatterAnim isActive={...} />
                    <AllReduceAnim isActive={isCommunicating} gpuPositions={validGpuCenters} centerPos={calculatedCenterPos} dataType={commDataType} containerOffset={{x:0, y:0}}/> // Placeholder using AllReduce visual
                )}


        </div>

        {/* Display KaTeX notation - Potentially integrate into OperationDetailsPanel later */}
        {/* <NotationDisplay notation={stepDetails?.notation || ''} description={stepDetails?.description || ''} /> */}
    </div>
  );
};

export default VisualizationArea;
