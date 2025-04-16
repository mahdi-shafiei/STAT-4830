import React from 'react'; import styles from './MemoryBar.module.css'; import { motion, AnimatePresence } from 'framer-motion'; import { GpuState } from '../../context/types'; // Assuming types are here
interface MemoryBarProps { type: 'Params' | 'Activations' | 'Gradients' | 'OptStates'; value: number; maxValue: number; shardDenom: number; isSharded: boolean; isTempFull?: boolean; gpuId: number; }
const MemoryBar: React.FC<MemoryBarProps> = ({ type, value, maxValue, shardDenom, isSharded, isTempFull, gpuId }) => { const persistentValue = value; const fullValue = isSharded && shardDenom > 0 ? persistentValue * shardDenom : persistentValue; const clampedPersistentValue = Math.min(persistentValue, maxValue); const clampedFullValue = Math.min(fullValue, maxValue); const persistentPercentage = maxValue > 0 ? (clampedPersistentValue / maxValue) * 100 : 0; const shardPercentage = isSharded && shardDenom > 0 ? (1 / shardDenom) * 100 : persistentPercentage; const solidBarPercentage = isSharded ? shardPercentage : persistentPercentage; const displayValue = isTempFull ? clampedFullValue : persistentValue; const displayMaxValue = maxValue; const shardInfo = isSharded && shardDenom > 0 ? `${gpuId + 1}/${shardDenom}` : ''; // Correct: use gpuId+1
    const labelText = `${type}${isSharded ? ` (Shard ${shardInfo})` : ''}:`; const barClass = styles[type.toLowerCase()] || ''; const tempFullVariants = { hidden: { opacity: 0 }, visible: { opacity: 0.35, transition: { duration: 0.15 } }, exit: { opacity: 0, transition: { duration: 0.1 } } };

    // Add a simple visual cue for sharding within the bar
    const renderShardDividers = (denom: number) => {
        if (!isSharded || denom <= 1) return null;
        return Array.from({ length: denom - 1 }).map((_, i) => (
            <div
                key={`divider-${type}-${gpuId}-${i}`}
                className={styles.shardDivider}
                style={{ left: `${((i + 1) / denom) * 100}%` }}
            />
        ));
    };

    return ( <div className={styles.memoryBarContainer} title={`Value: ${persistentValue.toFixed(1)} / Max: ${displayMaxValue}${isSharded ? ` (Shard ${shardInfo})` : ''}${isTempFull ? ' - Temp Full' : ''}`}> <span className={styles.memoryLabel}>{labelText}</span> <div className={styles.barBackground}> <motion.div className={`${styles.barForeground} ${barClass}`} initial={false} animate={{ width: `${solidBarPercentage}%` }} transition={{ duration: 0.3, ease: "easeInOut" }} /> <AnimatePresence> {isSharded && isTempFull && ( <motion.div key={`${type}-temp-overlay`} className={`${styles.barTempFullOverlay} ${barClass}`} variants={tempFullVariants} initial="hidden" animate="visible" exit="exit" style={{ width: '100%' }} /> )} </AnimatePresence> {renderShardDividers(shardDenom)} </div> </div> ); };
export default MemoryBar;
