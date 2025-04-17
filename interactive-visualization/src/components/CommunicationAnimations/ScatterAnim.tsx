import React from 'react';
import { motion, AnimatePresence, Variants } from 'framer-motion';
import styles from './CommAnimations.module.css';
import { Point } from '../../hooks/useGpuPositions';

interface ScatterAnimProps { isActive: boolean; sourcePos: Point; targetPositions: Point[]; dataType?: string; containerOffset: Point; }

export const ScatterAnim: React.FC<ScatterAnimProps> = ({ isActive, sourcePos, targetPositions, dataType = 'X', containerOffset }) => {
    if (!isActive || !sourcePos || targetPositions.length === 0) return null;

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
                     {adjustedTargets.map((target, i) => (
                        <motion.g key={`packet-group-${i}`}>
                            <motion.circle
                                className={`${styles.dataPacketCircle} ${styles.inputData}`}
                                custom={i}
                                variants={packetVariants}
                                initial="initial"
                                animate="animate"
                                exit="exit"
                                r={8}
                            />
                            {/* Add text inside circle - adjust position */}
                             <motion.text
                                 x={packetVariants.initial.cx} // Start position for text
                                 y={packetVariants.initial.cy}
                                 variants={{
                                     initial: { x: adjustedSource.x, y: adjustedSource.y, opacity: 0 },
                                     animate: { x: target.x, y: target.y + 3, opacity: 1, transition: packetVariants.animate(i).transition }, // Match packet anim
                                     exit: packetVariants.exit
                                 }}
                                 initial="initial" animate="animate" exit="exit"
                                 className={styles.packetText} // Add styling for text
                                 textAnchor="middle" // Center text
                             >
                                {`${dataType}${i}`} {/* Show shard index */}
                             </motion.text>
                        </motion.g>
                     ))}
                      {/* Label */}
                      <motion.text x={adjustedSource.x} y={adjustedSource.y - 15} className={styles.commLabel} initial={{opacity: 0}} animate={{opacity:1}} exit={{opacity:0}}>Scatter ({dataType})</motion.text>
                </motion.svg>
            )}
        </AnimatePresence>
    );
}; 