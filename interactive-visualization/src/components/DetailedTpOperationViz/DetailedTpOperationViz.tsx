import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import styles from './DetailedTpOperationViz.module.css';
import MathDisplay from '../MathDisplay/MathDisplay';

// --- Types ---
type Operation = 'ColumnLinear' | 'RowLinear';
type Phase = 'idle' | 'comm_input' | 'compute' | 'comm_output'; // Explicit phases
export interface TensorInfo { label: string; rows?: number | string; cols?: number | string; isSharded: 'col' | 'row' | false; numShards: number; }
export interface DetailedTpLinearOpVizProps {
    operation: Operation | null;
    phase: Phase;
    Ntp: number;
    inputTensor?: TensorInfo;
    weightTensor?: TensorInfo;
    outputTensor?: TensorInfo; // Final output after comms
    intermediateTensor?: TensorInfo; // Output before comms (e.g., Y_k)
    isActive?: boolean; // Added isActive prop
}

// --- Matrix Visual Component ---
const MatrixVisual: React.FC<{ tensorInfo: TensorInfo | undefined, isHighlighted?: boolean, matrixId: string }> =
 ({ tensorInfo, isHighlighted = false, matrixId }) => {
    if (!tensorInfo) return <div className={styles.matrixPlaceholder} />; // Placeholder if tensor info is missing
    const { label, rows = 'N', cols = 'M', isSharded, numShards } = tensorInfo;
    const width = isSharded === 'col' ? `${100 / numShards}%` : '100%';
    const height = isSharded === 'row' ? `${100 / numShards}%` : '100%';
    const shards = isSharded ? Array.from({length: numShards}) : [0]; // Use index for keys

    return (
        <motion.div layout key={matrixId} className={`${styles.matrixContainer} ${isHighlighted ? styles.highlighted : ''}`} title={`${rows}x${cols}${isSharded ? ` (${isSharded} sharded ${numShards})` : ''}`}>
            <div className={styles.matrixLabel}><MathDisplay texString={label} /></div>
            <div className={`${styles.matrix} ${isSharded ? styles[isSharded + 'Sharded'] : ''}`}>
                {shards.map((_, i) => (
                    <div key={i} className={styles.matrixShardVisual} style={isSharded === 'col' ? { width } : isSharded === 'row' ? { height } : {width: '100%', height: '100%'}} />
                ))}
            </div>
        </motion.div>
    );
};

// --- Main Component ---
export const DetailedTpLinearOpViz: React.FC<DetailedTpLinearOpVizProps> = ({ isActive, operation, phase, Ntp, inputTensor, weightTensor, outputTensor, intermediateTensor }) => {

    const getVisibility = (targetPhase: Phase | Phase[]): boolean => {
        if (!operation) return false;
        const phases = Array.isArray(targetPhase) ? targetPhase : [targetPhase];
        return phases.includes(phase);
    };

    const variants = { hidden: { opacity: 0, scale: 0.9 }, visible: { opacity: 1, scale: 1 }, exit: { opacity: 0, scale: 0.9 } };

    return (
        <div className={styles.detailedVizContainer}>
            <AnimatePresence mode="wait">
                {isActive && operation && phase !== 'idle' && (
                    <motion.div key={operation + phase} variants={variants} initial="hidden" animate="visible" exit="exit" transition={{ duration: 0.2 }}>
                        <div className={styles.operationTitle}>
                            TP {operation} - Phase: <span className={styles.phaseHighlight}>{phase}</span>
                        </div>
                        <div className={styles.computePhaseViz}>
                            {/* Input Tensor */}
                            <MatrixVisual matrixId={`input-${phase}`} tensorInfo={inputTensor} isHighlighted={getVisibility('compute')} />

                            {/* Operation Symbol */}
                             <span className={styles.opSymbol}>{getVisibility('compute') ? '✖️' : (getVisibility('comm_output') ? '➔' : '...' )}</span>

                            {/* Weight Tensor */}
                            <MatrixVisual matrixId={`weight-${phase}`} tensorInfo={weightTensor} isHighlighted={getVisibility('compute')} />

                            {/* Equals Symbol */}
                            <span className={styles.opSymbol}>=</span>

                            {/* Intermediate / Output Tensor */}
                            {getVisibility(['compute', 'comm_input']) && intermediateTensor && (
                                <MatrixVisual matrixId={`inter-${phase}`} tensorInfo={intermediateTensor} isHighlighted={getVisibility('compute')} />
                            )}
                            {getVisibility('comm_output') && outputTensor && (
                                <MatrixVisual matrixId={`output-${phase}`} tensorInfo={outputTensor} isHighlighted={true} />
                            )}
                             {/* Show placeholder if neither intermediate nor final output relevant for phase */}
                             {!getVisibility(['compute', 'comm_input', 'comm_output']) && <div className={styles.matrixPlaceholder} />}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}; 