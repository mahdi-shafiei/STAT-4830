import React from 'react';
import styles from './MemoryBar.module.css';

interface MemoryBarProps {
    type: 'Params' | 'Activations' | 'Gradients' | 'OptStates';
    value: number; // Current value
    maxValue: number; // Max value for scaling
    isSharded?: boolean;
    // Provide a default or ensure shardSize is always passed when isSharded is true
    shardSize?: number | string; // e.g., 1/N or just N
}

const MemoryBar: React.FC<MemoryBarProps> = ({ type, value, maxValue, isSharded, shardSize = 'N' }) => {
    // Ensure value doesn't exceed maxValue for visual representation
    const clampedValue = Math.min(value, maxValue);
    const percentage = maxValue > 0 ? (clampedValue / maxValue) * 100 : 0;
    const shardInfo = typeof shardSize === 'number' ? `1/${shardSize}` : shardSize;

    // Determine bar color based on type
    const barClass = styles[type.toLowerCase()] || '';

    return (
        <div className={styles.memoryBarContainer}>
             {/* Uses the corrected shardInfo variable */}
            <span className={styles.memoryLabel}>{type}{isSharded ? ` (Shard ${shardInfo})` : ''}: </span>
            <div className={styles.barBackground}>
                <div
                    className={`${styles.barForeground} ${barClass}`}
                    style={{ width: `${percentage}%` }}
                    title={`${value} / ${maxValue}`} // Show exact value on hover
                />
            </div>
        </div>
    );
};

export default MemoryBar;
