import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import styles from './TpLayerExecutionViz.module.css';
import MathDisplay from '../MathDisplay/MathDisplay';

export type TpOperationType = 'ColumnParallel' | 'RowParallelCompute' | 'RowParallelAllReduce' | 'LocalAttention' | 'Replicated' | 'Idle';

export interface TpStepInfo {
  operationType: TpOperationType;
  layerName?: string; // Optional for idle state
  inputDesc?: string;
  weightDesc?: string;
  outputDesc?: string; // Describes the *final* output of the operation/communication
  intermediateDesc?: string; // Describes the intermediate output (e.g., Z_k before AllReduce)
}

interface TpLayerExecutionVizProps {
  tpStepInfo: TpStepInfo | null;
  tpSize: number; // e.g., 2
  isActive: boolean;
}

// --- Animated Tensor Visual ---
interface AnimatedTensorVisualProps {
  label: string;
  isSharded: boolean;
  numShards: number;
  matrixId: string; // Unique key for animation presence
  isInput?: boolean;
  isOutput?: boolean;
  isWeight?: boolean;
  animate?: object; // Framer motion animate prop
  initial?: object | boolean;
  exit?: object;
}
const AnimatedTensorVisual: React.FC<AnimatedTensorVisualProps> =
  ({ label, isSharded, numShards, matrixId, isInput, isOutput, isWeight, animate, initial, exit }) => {
    const shardWidth = isSharded ? `${100 / numShards}%` : '100%';
    const shards = isSharded ? Array.from({ length: numShards }, (_, i) => i) : [0];
    const containerVariants = { hidden: { opacity: 0, scale: 0.8 }, visible: { opacity: 1, scale: 1 }, exit: { opacity: 0, scale: 0.8 } };

    return (
        <motion.div
            key={matrixId}
            className={`${styles.tensor} ${isInput ? styles.inputTensor : ''} ${isOutput ? styles.outputTensor : ''} ${isWeight ? styles.weightTensor : ''}`}
            variants={containerVariants}
            initial={initial === undefined ? "hidden" : initial}
            animate={animate || "visible"}
            exit={exit || "exit"}
            transition={{ duration: 0.3 }}
            layout // Animate layout changes
        >
            <div className={styles.tensorLabel}><MathDisplay texString={label} /></div>
            <motion.div className={styles.matrix} layout>
                {shards.map(shardIndex => (
                    <motion.div
                        key={shardIndex}
                        className={`${styles.matrixShard} ${isSharded ? styles.sharded : ''}`}
                        style={{ width: shardWidth }}
                        title={`Shard ${shardIndex + 1}/${numShards}`}
                        layout // Animate shard layout
                    />
                ))}
            </motion.div>
        </motion.div>
    );
};

// --- Main Component ---
const TpLayerExecutionViz: React.FC<TpLayerExecutionVizProps> = ({ tpStepInfo, tpSize, isActive }) => {

    const { operationType = 'Idle', layerName = '', inputDesc = '', weightDesc = '', outputDesc = '', intermediateDesc = '' } = tpStepInfo || {};

    // Define animation states for different operations
    let inputConfig = { label: inputDesc, isSharded: false, key: 'input' };
    let weightConfig = { label: weightDesc, isSharded: false, key: 'weight' };
    let intermediateConfig = { label: intermediateDesc || '', isSharded: false, key: 'inter' };
    let outputConfig = { label: outputDesc, isSharded: false, key: 'output' };
    let showIntermediate = false;
    let showOutput = true;
    let operationSymbol = 'Idle';
    let commSymbol: React.ReactNode = null;

    switch (operationType) {
        case 'ColumnParallel':
            inputConfig.isSharded = false;
            weightConfig.isSharded = true;
            showIntermediate = true;
            intermediateConfig = { label: intermediateDesc || outputDesc, isSharded: true, key: 'inter-col' }; // Output is sharded intermediate
            showOutput = false; // No final combined output in this step
            operationSymbol = '✖️'; // Multiply
            break;
        case 'RowParallelCompute':
            inputConfig.isSharded = true;
            weightConfig.isSharded = true;
            showIntermediate = true;
            intermediateConfig = { label: intermediateDesc || `Z_k`, isSharded: true, key: 'inter-row' }; // Partial output is sharded
            showOutput = false; // Final output comes after AllReduce
            operationSymbol = '✖️';
            break;
        case 'RowParallelAllReduce':
             inputConfig = { label: intermediateDesc || `Z_k`, isSharded: true, key: 'inter-row-in' }; // Input is the intermediate Z_k
             weightConfig = { label: `\\text{AllReduce}`, isSharded: false, key: 'comm-weight' }; // Show AllReduce op
             showIntermediate = false;
             showOutput = true; // Output is the final combined result
             outputConfig = { label: outputDesc || `A_{${layerName}}`, isSharded: false, key: 'output-row' };
             operationSymbol = '🔄'; // Communication symbol
             commSymbol = <div className={styles.commIndicator}>AllReduce</div>;
            break;
        case 'LocalAttention':
            inputConfig = { label: inputDesc || `Q_k,K_k,V_k`, isSharded: true, key: 'input-attn' };
            weightConfig = { label: weightDesc || `\\text{Attn Fn}`, isSharded: false, key: 'weight-attn' };
            showIntermediate = true;
            intermediateConfig = { label: intermediateDesc || `O_k`, isSharded: true, key: 'inter-attn' };
            showOutput = false;
            operationSymbol = '🧠'; // Attention symbol
            break;
         case 'Replicated':
            inputConfig.isSharded = false;
            weightConfig = { label: weightDesc || `w_{${layerName}}`, isSharded: false, key: 'weight-rep' };
            showIntermediate = false;
            showOutput = true;
            outputConfig = { label: outputDesc || `A_{${layerName}}`, isSharded: false, key: 'output-rep' };
            operationSymbol = '⚙️'; // Generic processing
            break;
        default: // Idle
             showOutput = false; showIntermediate = false;
             inputConfig = { label: 'N/A', isSharded: false, key: 'input-idle'};
             weightConfig = { label: 'N/A', isSharded: false, key: 'weight-idle'};
             operationSymbol = '-';
             break;
    }

    const vizBoxVariants = { hidden: { opacity: 0.5, height: 0 }, visible: { opacity: 1, height: 'auto' }, exit: { opacity: 0, height: 0 } };

    return (
        <div className={styles.vizContainer}> {/* Added container for positioning */}
            <AnimatePresence>
                {isActive && operationType !== 'Idle' && (
                    <motion.div
                        key={operationType + layerName}
                        className={styles.vizBox}
                        variants={vizBoxVariants}
                        initial="hidden"
                        animate="visible"
                        exit="exit"
                        transition={{ duration: 0.3 }}
                    >
                        <div className={styles.vizTitle}>{layerName} - TP: {operationType}</div>
                        <motion.div className={styles.vizContent} layout>
                            {/* Input Tensor */}
                            <AnimatedTensorVisual
                                matrixId={inputConfig.key}
                                label={inputConfig.label}
                                isSharded={inputConfig.isSharded}
                                numShards={tpSize}
                                isInput={true}
                                initial={false} // Let AnimatePresence handle initial
                            />

                            <motion.div layout className={styles.operationSymbol}>{operationSymbol}</motion.div>

                            {/* Weight Tensor */}
                            <AnimatedTensorVisual
                                matrixId={weightConfig.key}
                                label={weightConfig.label}
                                isSharded={weightConfig.isSharded}
                                numShards={tpSize}
                                isWeight={true}
                                initial={false}
                            />

                            {showIntermediate && (
                                <>
                                    <motion.div layout className={styles.operationSymbol}>=</motion.div>
                                    {/* Intermediate Output Tensor */}
                                    <AnimatedTensorVisual
                                         matrixId={intermediateConfig.key}
                                         label={intermediateConfig.label}
                                         isSharded={intermediateConfig.isSharded}
                                         numShards={tpSize}
                                         isOutput={true} // It's an output of this *stage*
                                         initial={false}
                                    />
                                 </>
                            )}
                             {showOutput && !showIntermediate && ( // Only show final output if no intermediate shown
                                 <>
                                     <motion.div layout className={styles.operationSymbol}>=</motion.div>
                                     {/* Final Output Tensor */}
                                     <AnimatedTensorVisual
                                         matrixId={outputConfig.key}
                                         label={outputConfig.label}
                                         isSharded={outputConfig.isSharded}
                                         numShards={tpSize}
                                         isOutput={true}
                                         initial={false}
                                     />
                                 </>
                             )}
                        </motion.div>
                        {commSymbol}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default TpLayerExecutionViz; 