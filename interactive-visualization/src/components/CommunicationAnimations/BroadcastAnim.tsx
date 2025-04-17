import React from 'react';
import { motion, AnimatePresence, Variants } from 'framer-motion';
import styles from './CommAnimations.module.css';
import { Point } from '../../hooks/useGpuPositions';

interface BroadcastAnimProps { isActive: boolean; sourcePos: Point; targetPositions: Point[]; dataType?: string; containerOffset: Point; }

export const BroadcastAnim: React.FC<BroadcastAnimProps> = ({ isActive, sourcePos, targetPositions, dataType = 'X', containerOffset }) => {
    if (!isActive || !sourcePos || targetPositions.length === 0) return null;

    // Adjust positions relative to the SVG container's top-left
    const adjustedSource = { x: sourcePos.x + containerOffset.x, y: sourcePos.y + containerOffset.y };
    const adjustedTargets = targetPositions.map(p => ({ x: p.x + containerOffset.x, y: p.y + containerOffset.y }));

    const packetVariants: Variants = {
        initial: { cx: adjustedSource.x, cy: adjustedSource.y, opacity: 0, scale: 0.5 },
        animate: (i: number) => ({
            cx: adjustedTargets[i].x,
            cy: adjustedTargets[i].y,
            opacity: 1,
            scale: 1,
            transition: { duration: 0.5, delay: 0.1 + i * 0.05, ease: "easeInOut" }
        }),
        exit: { opacity: 0, scale: 0.5, transition: { duration: 0.2 } }
    };
     const lineVariants: Variants = { initial: { pathLength: 0, opacity: 0 }, animate: { pathLength: 1, opacity: 0.6, transition: { duration: 0.4, delay: 0.1 } }, exit: { opacity: 0 } };

    return (
        <AnimatePresence>
            {isActive && (
                <motion.svg className={styles.commSvgOverlay}>
                    {/* Lines */}
                    {adjustedTargets.map((target, i) => (
                         <motion.path
                            key={`line-${i}`}
                            d={`M ${adjustedSource.x} ${adjustedSource.y} L ${target.x} ${target.y}`}
                            className={styles.commLine}
                            variants={lineVariants} initial="initial" animate="animate" exit="exit"
                        />
                    ))}
                     {/* Packets */}
                     {adjustedTargets.map((_, i) => (
                        <motion.circle
                            key={`packet-${i}`}
                            className={`${styles.dataPacketCircle} ${styles.inputData}`} // Style as circle
                            custom={i}
                            variants={packetVariants}
                            initial="initial"
                            animate="animate"
                            exit="exit"
                            r={8} // Radius of circle packet
                        />
                     ))}
                      {/* Label */}
                      <motion.text x={adjustedSource.x} y={adjustedSource.y - 15} className={styles.commLabel} initial={{opacity: 0}} animate={{opacity:1}} exit={{opacity:0}}>Broadcast ({dataType})</motion.text>
                </motion.svg>
            )}
        </AnimatePresence>
    );
}; 