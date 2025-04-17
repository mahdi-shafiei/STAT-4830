import React from 'react';
import { motion } from 'framer-motion';
import styles from './TpLayerExecutionViz.module.css';
import MathDisplay from '../MathDisplay/MathDisplay'; // Assuming MathDisplay exists

// Define types for props
export type TpOperationType = 'ColumnParallel' | 'RowParallel' | 'LocalAttention' | 'Replicated';

export interface TpStepInfo { // Export the interface
  operationType: TpOperationType;
  layerName: string;
  inputDesc?: string; // e.g., "X (Full)" or "A_prev_k (Sharded)"
  weightDesc?: string; // e.g., "W = [W_0 | W_1]" or "W = [W_0^T ; W_1^T]^T"
  outputDesc?: string; // e.g., "Y_k = X W_k" or "Y = AllReduce(Y_k)"
  isCommunicating?: boolean; // True during AllReduce phase of RowParallel
}

interface TpLayerExecutionVizProps {
  tpStepInfo: TpStepInfo | null;
  tpSize: number; // e.g., 2
  isActive: boolean; // Controls overall visibility
}

// Simple visual representation of a tensor/matrix
const TensorVisual: React.FC<{ label: string; sharded?: boolean; numShards?: number; highlightShard?: number; isInput?: boolean; isOutput?: boolean }> =
 ({ label, sharded = false, numShards = 1, highlightShard = -1, isInput = false, isOutput = false }) => {
    const shardWidth = sharded ? `${100 / numShards}%` : '100%';
    const shards = sharded ? Array.from({ length: numShards }, (_, i) => i) : [0];

    return (
        <div className={`${styles.tensor} ${isInput ? styles.inputTensor : ''} ${isOutput ? styles.outputTensor : ''}`}>
            <div className={styles.tensorLabel}>{label ? <MathDisplay texString={label} /> : null}</div> {/* Conditionally render MathDisplay */}
            <div className={styles.matrix}>
                {shards.map(shardIndex => (
                    <div
                        key={shardIndex}
                        className={`${styles.matrixShard} ${sharded ? styles.sharded : ''} ${highlightShard === shardIndex ? styles.highlightedShard : ''}`}
                        style={{ width: shardWidth }}
                        title={`Shard ${shardIndex + 1}/${numShards}`}
                    >
                        {/* Optionally add internal pattern */}
                    </div>
                ))}
            </div>
        </div>
    );
};


const TpLayerExecutionViz: React.FC<TpLayerExecutionVizProps> = ({ tpStepInfo, tpSize, isActive }) => {
    if (!isActive || !tpStepInfo) {
        return null;
    }

    const { operationType, layerName, inputDesc, weightDesc, outputDesc, isCommunicating } = tpStepInfo;

    const variants = {
        hidden: { opacity: 0, y: -20 },
        visible: { opacity: 1, y: 0, transition: { duration: 0.3 } },
        exit: { opacity: 0, y: 20, transition: { duration: 0.2 } }
    };

    // --- Determine Visual Layout based on Operation ---
    let inputTensorLabel = inputDesc || '';
    let weightTensorLabel = weightDesc || '';
    let outputTensorLabel = outputDesc || '';
    let showInputShards = false;
    let showWeightShards = false;
    let showOutputShards = false;

    if (operationType === 'ColumnParallel') {
        showWeightShards = true;
        showOutputShards = true;
    } else if (operationType === 'RowParallel') {
        showInputShards = true;
        showWeightShards = true;
        // Output is initially sharded conceptually, then AllReduced to full
        showOutputShards = isCommunicating ? false : true; // Show sharded partial output before comms
        outputTensorLabel = isCommunicating ? `Y = \\text{AllReduce}(Y_k)` : outputDesc || '';
    } else if (operationType === 'LocalAttention') {
        inputTensorLabel = `Q_k, K_k, V_k`; // Simplified input
        showInputShards = true; // QKV inputs are sharded
        weightTensorLabel = `\\text{Attn}(Q_k, K_k, V_k)`; // Operation as "weight"
        showWeightShards = false;
        outputTensorLabel = `O_k`; // Output is sharded
        showOutputShards = true;
    } else if (operationType === 'Replicated') {
        // No sharding visible
    }


    return (
        <motion.div
            key={layerName + operationType + (isCommunicating ? '-comm' : '-comp')} // Ensure re-render on change, incl. comms phase
            className={styles.vizOverlay}
            variants={variants}
            initial="hidden"
            animate="visible"
            exit="exit"
        >
            <div className={styles.vizBox}>
                <div className={styles.vizTitle}>{layerName} - TP: {operationType} {isCommunicating ? '(Communicate)' : '(Compute)'}</div>
                <div className={styles.vizContent}>
                    {/* Input */}
                    <TensorVisual label={inputTensorLabel} sharded={showInputShards} numShards={tpSize} isInput={true} />

                    <div className={styles.operationSymbol}>{isCommunicating ? '🔄' : '✖️'}</div>

                    {/* Weights */}
                    <TensorVisual label={weightTensorLabel} sharded={showWeightShards} numShards={tpSize} />

                     <div className={styles.operationSymbol}>=</div>

                     {/* Output */}
                    <TensorVisual label={outputTensorLabel} sharded={showOutputShards && !isCommunicating} numShards={tpSize} isOutput={true} />
                </div>
                 {isCommunicating && <div className={styles.commIndicator}>Executing AllReduce...</div>}
            </div>
        </motion.div>
    );
};

export default TpLayerExecutionViz; 