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

  // --- Logic for TP Visualization ---
  const showTpViz = strategy === 'tp' && stepDetails && (stepDetails.type === 'COMPUTE' || stepDetails.type === 'COMM') && stepDetails.layer && stepDetails.tpExecutionType;

  const tpVizInfo: TpStepInfo | null = showTpViz && stepDetails ? { // Ensure stepDetails exists
      operationType: stepDetails.tpExecutionType as TpOperationType, // Cast needed
      layerName: stepDetails.layer as string,
      inputDesc: stepDetails.inputDesc,
      weightDesc: stepDetails.weightDesc,
      outputDesc: stepDetails.outputDesc,
      // Determine if the specific TP step involves communication (currently only RowParallel AllReduce)
      isCommunicating: stepDetails.type === 'COMM' && stepDetails.operation === 'AllReduce' && stepDetails.tpExecutionType === 'RowParallel'
  } : null;

  // Simple layout logic: place GPUs side-by-side
  return (
    <div className={styles.visualizationArea}>
        {/* Layer Icons (Top) - If you add them, they would go here */}
        {/* Placeholder for layer icons area */} 

        {/* --- TP Execution Visualization --- */}
        <div className={styles.tpVizContainer}> {/* Added container for positioning */} 
          <AnimatePresence>
              {showTpViz && tpVizInfo && (
                  <TpLayerExecutionViz
                      key={stepDetails?.step} // Key ensures update on step change
                      tpStepInfo={tpVizInfo}
                      tpSize={2} // Ntp = 2 fixed for this strategy
                      isActive={true}
                  />
              )}
          </AnimatePresence>
        </div>

        <div className={styles.gpuContainer}>
             <AnimatePresence>
                {gpuStates.map((gpuState, index) => {
                    // Determine sharding flags based on strategy
                    const isParamsSharded = strategy === 'fsdp' || strategy === 'tp';
                    const isGradsSharded = strategy === 'fsdp'; // TP grads sharding handled within MemoryBar logic maybe
                    const isOptStatesSharded = strategy === 'fsdp' || strategy === 'tp';

                    return (
                        <Gpu
                            key={gpuState.id}
                            {...gpuState} // Spread existing state
                            numGpusInGroup={numGpus} // Pass total GPU count for context
                            isParamsSharded={isParamsSharded}
                            isGradsSharded={isGradsSharded} // Pass grad sharding flag
                            isOptStatesSharded={isOptStatesSharded}
                            currentStepDetails={stepDetails} // Pass current step details
                        />
                    );
                })}
            </AnimatePresence>
        </div>

      {/* --- Communication Arrow --- */}
      {/* Hide generic arrow if TP viz is showing the communication or if not communicating */}
      <AnimatePresence>
          {isCommunicating && numGpus > 1 && !showTpViz && commDataType && commOperation && (
            <CommunicationArrow
              numGpus={numGpus}
              dataType={commDataType}
              operation={commOperation}
              strategy={strategy}
            />
          )}
       </AnimatePresence>

      {/* Display KaTeX notation - Potentially integrate into OperationDetailsPanel later */}
      {/* <NotationDisplay notation={stepDetails?.notation || ''} description={stepDetails?.description || ''} /> */}
    </div>
  );
};

export default VisualizationArea;
