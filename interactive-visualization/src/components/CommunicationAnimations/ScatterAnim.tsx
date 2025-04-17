import React from 'react';
import { motion, AnimatePresence, Variants } from 'framer-motion';
import styles from './CommAnimations.module.css';
import { Point } from '../../hooks/useGpuPositions';

interface ScatterAnimProps { isActive: boolean; sourcePos: Point | null; targetPositions: (Point | null)[]; dataType?: string | null; containerOffset?: Point; }

export const ScatterAnim: React.FC<ScatterAnimProps> = ({ isActive, sourcePos, targetPositions, dataType, containerOffset = {x:0, y:0} }) => {
    const validTargets = targetPositions.filter((p): p is Point => p !== null);
    if (!isActive || !sourcePos || validTargets.length === 0) return null;

    const adjustedSource = { x: sourcePos.x + containerOffset.x, y: sourcePos.y + containerOffset.y };
    const adjustedTargets = validTargets.map(p => ({ x: p.x + containerOffset.x, y: p.y + containerOffset.y }));
    const duration = 0.5;
    const delay = 0.1;

    const packetVariants: Variants = {
         initial: { cx: adjustedSource.x, cy: adjustedSource.y, opacity: 0, scale: 0.5 },
         animate: (i: number) => ({
             cx: adjustedTargets[i].x,
             cy: adjustedTargets[i].y,
             opacity: 1,
             scale: 1,
             transition: { duration: duration, delay: delay + i * 0.05, ease: "easeInOut" }
         }),
         exit: { opacity: 0, scale: 0.5, transition: { duration: 0.2 } }
    };
    const textVariants: Variants = {
        initial: { x: adjustedSource.x, y: adjustedSource.y + 3, opacity: 0 },
        animate: (i: number) => ({ x: adjustedTargets[i].x, y: adjustedTargets[i].y + 3, opacity: 1, transition: packetVariants.animate(i).transition }),
        exit: { opacity: 0 }
    }
     const lineVariants: Variants = { initial: { pathLength: 0, opacity: 0 }, animate: { pathLength: 1, opacity: 0.6, transition: { duration: duration * 0.8, delay: delay } }, exit: { opacity: 0 } };
     const baseDataType = dataType || 'X';
     const labelText = `Scatter${dataType ? ` (${dataType})` : ''}`;

    return (
        <AnimatePresence>
            {isActive && (
                 <motion.svg className={styles.commSvgOverlay} key="scatter-svg">
                    {/* Lines */}
                    {adjustedTargets.map((target, i) => (
                         <motion.path
                            key={`line-${i}`}
                            d={`M ${adjustedSource.x} ${adjustedSource.y} L ${target.x} ${target.y}`}
                            className={styles.commLine}
                            variants={lineVariants} initial="initial" animate="animate" exit="exit"
                        />
                    ))}
                     {/* Packets with Text */}
                     {adjustedTargets.map((_, i) => (
                        <motion.g key={`packet-group-${i}`}>
                            <motion.circle
                                className={`${styles.dataPacketCircle} ${styles.inputData}`}
                                custom={i}
                                variants={packetVariants}
                                initial="initial"
                                animate="animate"
                                exit="exit"
                                r={10} // Slightly larger to fit text
                            />
                             <motion.text
                                 custom={i}
                                 variants={textVariants}
                                 initial="initial" animate="animate" exit="exit"
                                 className={styles.packetText}
                                 textAnchor="middle"
                             >
                                {`${baseDataType[0]}${i}`} {/* Show shard index e.g., X0, X1 */}
                             </motion.text>
                        </motion.g>
                     ))}
                      {/* Label */}
                      <motion.text x={adjustedSource.x} y={adjustedSource.y - 15} className={styles.commLabel} initial={{opacity: 0}} animate={{opacity:1, transition:{delay: delay}}} exit={{opacity:0}}>{labelText}</motion.text>
                </motion.svg>
            )}
        </AnimatePresence>
    );
}; 