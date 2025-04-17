import React from 'react';
import Gpu from '../Gpu/Gpu';
import CommunicationArrow from '../CommunicationArrow/CommunicationArrow';
import NotationDisplay from '../MathDisplay/MathDisplay';
import styles from './VisualizationArea.module.css';
import { useSimulation } from '../../context/SimulationContext';
import { AnimatePresence } from 'framer-motion';
import TpLayerExecutionViz from '../TpLayerExecutionViz/TpLayerExecutionViz';
import type { TpStepInfo, TpOperationType } from '../TpLayerExecutionViz/TpLayerExecutionViz';

const VisualizationArea: React.FC = () => {
  const { gpuStates, stepDetails, numGpus, strategy } = useSimulation();

  // Determine if communication is happening based on stepDetails
  const isCommunicating = stepDetails?.type === 'COMM';
  const commDataType = stepDetails?.dataType;
  const commOperation = stepDetails?.operation;
  const commType = stepDetails?.operation; // Using commType for clarity in arrow logic

  // --- Logic for TP Visualization ---
  const showTpViz = strategy === 'tp' && stepDetails && (stepDetails.type === 'COMPUTE' || stepDetails.type === 'COMM') && stepDetails.tpExecutionType;

  const tpVizInfo: TpStepInfo | null = showTpViz && stepDetails ? {
      // Determine operationType based on step details
      operationType: (stepDetails.type === 'COMM' && stepDetails.operation === 'AllReduce')
           ? 'RowParallelAllReduce'
           : (stepDetails.tpExecutionType as TpOperationType) || 'Idle', // Default to Idle if type missing
      layerName: stepDetails.layer || '',
      inputDesc: stepDetails.inputDesc,
      weightDesc: stepDetails.weightDesc,
      // Use intermediateDesc for RowParallel compute output notation passed from context
      intermediateDesc: (stepDetails.tpExecutionType === 'RowParallel' && stepDetails.type === 'COMPUTE') ? stepDetails.outputDesc : stepDetails.intermediateDesc,
      outputDesc: stepDetails.outputDesc, // Final output description
      // Remove isCommunicating, handled by operationType
  } : null;

  // Simple layout logic: place GPUs side-by-side
  return (
    <div className={styles.visualizationArea}>
        {/* Layer Icons (Top) - If you add them, they would go here */}
        {/* Placeholder for layer icons area */}

        {/* --- TP Execution Visualization --- */}
        <div className={styles.tpVizContainer}> {/* Keep existing container */} 
          <AnimatePresence>
              {/* Ensure tpVizInfo is passed correctly */} 
              {showTpViz && tpVizInfo && (
                  <TpLayerExecutionViz
                      key={stepDetails?.step + (tpVizInfo.operationType || 'idle')} // More robust key
                      tpStepInfo={tpVizInfo}
                      tpSize={numGpus} // Ntp = numGpus for TP strategy (assuming tpSize=numGpus)
                      isActive={true} // Controlled by AnimatePresence now
                  />
              )}
          </AnimatePresence>
        </div>

        {/* --- GPU Container --- */}
        <div className={styles.gpuContainer}>
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

      {/* --- Communication Arrow --- */}
      {/* Hide for TP comms handled by TpLayerExecutionViz */} 
      <AnimatePresence>
          {isCommunicating && numGpus > 1 && commDataType && commOperation && !showTpViz && // Hide if TP viz is showing
                // !(strategy === 'tp' && commType === 'AllReduce' && commDataType === 'Activations') && // Specifically hide TP Act AllReduce (redundant if !showTpViz covers it)
                ( <CommunicationArrow
                    numGpus={numGpus}
                    dataType={commDataType}
                    operation={commOperation}
                    strategy={strategy}
                  />
                )
          }
       </AnimatePresence>

      {/* Display KaTeX notation - Potentially integrate into OperationDetailsPanel later */}
      {/* <NotationDisplay notation={stepDetails?.notation || ''} description={stepDetails?.description || ''} /> */}
    </div>
  );
};

export default VisualizationArea;
