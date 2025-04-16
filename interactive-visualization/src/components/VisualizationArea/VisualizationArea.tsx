import React from 'react';
import Gpu from '../Gpu/Gpu';
import CommunicationArrow from '../CommunicationArrow/CommunicationArrow';
import NotationDisplay from '../MathDisplay/MathDisplay';
import styles from './VisualizationArea.module.css';
import { useSimulation } from '../../context/SimulationContext';
import { AnimatePresence } from 'framer-motion';

const VisualizationArea: React.FC = () => {
  const { gpuStates, stepDetails, numGpus, strategy } = useSimulation();

  // Determine if communication is happening based on stepDetails
  const isCommunicating = stepDetails?.type === 'COMM';
  const commDataType = stepDetails?.dataType;
  const commOperation = stepDetails?.operation;

  // Simple layout logic: place GPUs side-by-side
  return (
    <div className={styles.visualizationArea}>
        <div className={styles.gpuContainer}>
             <AnimatePresence>
                {gpuStates.map((gpuState, index) => {
                    // Determine sharding flags based on strategy
                    const isParamsSharded = strategy === 'fsdp' || strategy === 'tp';
                    const isGradsSharded = strategy === 'fsdp'; // Simplify: Only FSDP explicitly shards grads this way in viz for now
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

      {/* Conditional rendering for communication arrows */}
      {isCommunicating && numGpus > 1 && (
        <CommunicationArrow
          numGpus={numGpus}
          dataType={commDataType}
          operation={commOperation}
          strategy={strategy}
        />
      )}

      {/* Display KaTeX notation */}
      <NotationDisplay notation={stepDetails?.notation || ''} description={stepDetails?.description || ''} />
    </div>
  );
};

export default VisualizationArea;
