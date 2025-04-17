import React, { useRef, useMemo } from 'react';
import Gpu from '../Gpu/Gpu';
import PipelineStageIndicator from '../PipelineStageIndicator/PipelineStageIndicator';
import { DetailedTpLinearOpViz } from '../DetailedTpOperationViz/DetailedTpOperationViz';
import { BroadcastAnim } from '../CommunicationAnimations/BroadcastAnim';
import { ScatterAnim } from '../CommunicationAnimations/ScatterAnim';
import { AllReduceAnim } from '../CommunicationAnimations/AllReduceAnim';
import { AllGatherAnim } from '../CommunicationAnimations/AllGatherAnim';
// Assume ReduceScatterAnim will be created later or use AllReduce as placeholder
import styles from './VisualizationArea.module.css';
import { useSimulation, GpuState, StepDetail } from '../../context/SimulationContext'; // Import GpuState type
import { useGpuPositions, Point } from '../../hooks/useGpuPositions';
import { AnimatePresence } from 'framer-motion';
import type { OperationPhase, TensorInfo } from '../DetailedTpOperationViz/DetailedTpOperationViz'; // Import types

const MODEL_LAYERS = ['Embed', 'MHA', 'FFN', 'LN', 'Output'];

const VisualizationArea: React.FC = () => {
    const { numGpus, gpuStates, currentStep, stepDetails, strategy } = useSimulation();
    const gpuContainerRef = useRef<HTMLDivElement>(null);

    // --- Position Calculation ---
    // Get raw positions and container rect relative to viewport
    const { positions: gpuDomInfos, containerRect: vizContainerRect } = useGpuPositions(gpuContainerRef, numGpus, currentStep);

    // Calculate center positions relative to the container for animations
    const gpuCenters = useMemo(() => {
        if (!vizContainerRect) return [];
        return gpuDomInfos.map(info => ({
            x: info.center.x, // Already relative to container in hook v3
            y: info.center.y
        }));
    }, [gpuDomInfos, vizContainerRect]);

    // Calculate a center point for animations like AllReduce, relative to container
    const centerPos = useMemo(() => {
        if (!vizContainerRect || gpuCenters.length === 0) return { x: 300, y: 150 }; // Fallback
         // Find bounding box of GPUs to center within them
         let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
         gpuCenters.forEach(p => {
             minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
             minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
         });
         const centerX = (minX + maxX) / 2;
         const centerY = (minY + maxY) / 2 - 80; // Place above GPU centers
         return { x: centerX, y: Math.max(50, centerY) }; // Ensure not too high
    }, [gpuCenters, vizContainerRect]);

     // --- Determine Communication State ---
     const isCommunicating = stepDetails?.type === 'COMM';
     const commOperation = isCommunicating ? stepDetails.operation : undefined;
     const commDataType = isCommunicating ? stepDetails.dataType : undefined;

     // --- Determine Detailed TP Viz State ---
     let detailedTpOperation: 'ColumnLinear' | 'RowLinear' | null = null;
     let detailedTpPhase: OperationPhase = 'idle';
     let inputTensor = stepDetails?.inputTensor;
     let weightTensor = stepDetails?.weightTensor;
     let outputTensor = stepDetails?.outputTensor;
     let intermediateTensor = stepDetails?.intermediateTensor;

     if (strategy === 'tp' && stepDetails?.tpExecutionType) {
         detailedTpPhase = stepDetails.phase || 'compute';
         const execType = stepDetails.tpExecutionType;
         if (execType === 'ColumnLinear') detailedTpOperation = 'ColumnLinear';
         else if (execType === 'RowParallel') detailedTpOperation = 'RowLinear';
         // Update tensor visibility based on phase for RowLinear AllReduce
          if (detailedTpOperation === 'RowLinear' && detailedTpPhase === 'comm_output') {
               inputTensor = stepDetails.intermediateTensor; // Input to AllReduce is the intermediate Zk
               outputTensor = stepDetails.outputTensor;
               weightTensor = undefined;
               intermediateTensor = undefined;
          } else if (detailedTpOperation === 'RowLinear' && detailedTpPhase === 'compute') {
               outputTensor = undefined; // No final output yet
          } else if (detailedTpOperation === 'ColumnLinear' && detailedTpPhase === 'compute') {
                outputTensor = undefined; // No final output yet (using intermediate)
          }
     }
     const showDetailedTpViz = detailedTpOperation !== null && detailedTpPhase !== 'idle';


    return (
        // Use container class matching App.module.css
        <div className={styles.visualizationAreaContainer}>
            {/* Pipeline Stage Indicator */}
            <PipelineStageIndicator layers={MODEL_LAYERS} currentLayer={stepDetails?.layer} />

            {/* Detailed TP Linear Op Viz (renders in flow now) */}
            <AnimatePresence>
               {showDetailedTpViz && (
                    <DetailedTpLinearOpViz
                        key={detailedTpOperation! + detailedTpPhase} // Use non-null assertion
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

            {/* GPU Container (attach ref) */}
            <div ref={gpuContainerRef} className={styles.gpuContainer} style={{ gridTemplateColumns: `repeat(${Math.min(numGpus, 4)}, 1fr)` }}>
                {gpuStates.map((gpuState: GpuState) => {
                    const isParamsSharded = strategy === 'fsdp' || strategy === 'tp';
                    const isGradsSharded = strategy === 'fsdp'; // Simplify viz
                    const isOptStatesSharded = strategy === 'fsdp' || strategy === 'tp';
                    return (
                        <Gpu
                            key={gpuState.id}
                            {...gpuState}
                            numGpusInGroup={numGpus}
                            isParamsSharded={isParamsSharded}
                            isGradsSharded={isGradsSharded}
                            isOptStatesSharded={isOptStatesSharded}
                            currentStepDetails={stepDetails}
                            strategy={strategy}
                        />
                    );
                })}
            </div>

            {/* Communication Animations (Absolute Positioned) */}
            {/* The SVG overlay needs to be sized correctly relative to the viz area */}
            <div className={styles.communicationAnimationContainer}>
                {/* Render animations based on current step and strategy */}
                {/* Pass valid center positions and container offset = 0,0 since SVG is positioned relative to this container */}
                 {isCommunicating && commOperation === 'Broadcast' && strategy === 'tp' && (
                     <BroadcastAnim isActive={true} sourcePos={centerPos} targetPositions={gpuCenters} dataType={commDataType} containerOffset={{x:0, y:0}}/>
                 )}
                  {isCommunicating && commOperation === 'Scatter' && strategy === 'tp' && (
                     <ScatterAnim isActive={true} sourcePos={centerPos} targetPositions={gpuCenters} dataType={commDataType} containerOffset={{x:0, y:0}}/>
                 )}
                  {isCommunicating && commOperation === 'AllReduce' && (strategy === 'tp' || strategy === 'dp') && (
                     <AllReduceAnim isActive={true} gpuPositions={gpuCenters} centerPos={centerPos} dataType={commDataType} containerOffset={{x:0, y:0}}/>
                 )}
                  {isCommunicating && commOperation === 'AllGather' && (strategy === 'tp' || strategy === 'fsdp') && ( // Added FSDP
                     <AllGatherAnim isActive={true} gpuPositions={gpuCenters} dataType={commDataType} containerOffset={{x:0, y:0}}/>
                 )}
                  {isCommunicating && commOperation === 'ReduceScatter' && strategy === 'fsdp' && ( // Added FSDP
                      // Use AllReduce visual as placeholder for ReduceScatter
                     <AllReduceAnim isActive={true} gpuPositions={gpuCenters} centerPos={centerPos} dataType={commDataType} containerOffset={{x:0, y:0}}/>
                  )}
            </div>
        </div>
    );
};
export default VisualizationArea;
