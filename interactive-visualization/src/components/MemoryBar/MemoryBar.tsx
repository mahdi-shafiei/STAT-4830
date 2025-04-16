import React from 'react';
import styles from './MemoryBar.module.css';
import { motion, AnimatePresence } from 'framer-motion';

interface MemoryBarProps {
    type: 'Params' | 'Activations' | 'Gradients' | 'OptStates';
    value: number; // Current persistent value (sharded or full)
    maxValue: number; // Max value of the *full* tensor
    shardDenom: number; // Denominator for sharding (e.g., numGpus for FSDP)
    isSharded: boolean; // Is this memory type currently sharded?
    isTempFull?: boolean; // Is it temporarily gathered to full size?
    gpuId: number; // ID of the GPU this bar belongs to
}

const MemoryBar: React.FC<MemoryBarProps> = ({
    type, value, maxValue, shardDenom, isSharded, isTempFull, gpuId
}) => {
    const persistentValue = value;
    const fullValue = isSharded && shardDenom > 0 ? persistentValue * shardDenom : persistentValue;
    const clampedPersistentValue = Math.min(persistentValue, maxValue);
    const clampedFullValue = Math.min(fullValue, maxValue);
    const persistentPercentage = maxValue > 0 ? (clampedPersistentValue / maxValue) * 100 : 0;
    const shardPercentage = isSharded && shardDenom > 0 ? (1 / shardDenom) * 100 : persistentPercentage;
    const solidBarPercentage = isSharded ? shardPercentage : persistentPercentage;
    const displayValue = isTempFull ? clampedFullValue : persistentValue; // Show persistent value even when temp full for clarity
    const displayMaxValue = maxValue;

    // FIX: Display correct shard index k/N (using gpuId + 1)
    const shardInfo = isSharded && shardDenom > 0 ? `${gpuId + 1}/${shardDenom}` : '';
    const labelText = `${type}${isSharded ? ` (Shard ${shardInfo})` : ''}:`;

    const barClass = styles[type.toLowerCase()] || '';
    const tempFullVariants = { hidden: { opacity: 0 }, visible: { opacity: 0.35, transition: { duration: 0.15 } }, exit: { opacity: 0, transition: { duration: 0.1 } } };

    return (
        <div className={styles.memoryBarContainer} title={`Value: ${persistentValue.toFixed(1)} / Max: ${displayMaxValue}${isSharded ? ` (Shard ${shardInfo})` : ''}${isTempFull ? ' - Temp Full' : ''}`}>
            <span className={styles.memoryLabel}>{labelText}</span>
            <div className={styles.barBackground}>
                <motion.div
                    className={`${styles.barForeground} ${barClass}`}
                    initial={false}
                    animate={{ width: `${solidBarPercentage}%` }}
                    transition={{ duration: 0.3, ease: "easeInOut" }}
                />
                <AnimatePresence>
                    {isSharded && isTempFull && (
                        <motion.div
                            key={`${type}-temp-overlay`}
                            className={`${styles.barTempFullOverlay} ${barClass}`}
                            variants={tempFullVariants}
                            initial="hidden"
                            animate="visible"
                            exit="exit"
                            style={{ width: '100%' }}
                        />
                    )}
                 </AnimatePresence>
            </div>
        </div>
    );
};
export default MemoryBar;
