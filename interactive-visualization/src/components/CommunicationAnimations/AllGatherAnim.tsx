import React from 'react';
import { motion, AnimatePresence, Variants } from 'framer-motion';
import styles from './CommAnimations.module.css';
import { Point } from '../../hooks/useGpuPositions';

interface AllGatherAnimProps { isActive: boolean; gpuPositions: Point[]; dataType?: string; containerOffset: Point; }

export const AllGatherAnim: React.FC<AllGatherAnimProps> = ({ isActive, gpuPositions, dataType = 'Param', containerOffset }) => {
     if (!isActive || gpuPositions.length === 0) return null;

     const adjustedGpus = gpuPositions.map(p => ({ x: p.x + containerOffset.x, y: p.y + containerOffset.y }));
     const numGpus = adjustedGpus.length;
     const duration = 0.5;
     const delay = 0.1;

     // Each GPU sends its packet to all other GPUs (including itself conceptually)
    const packetVariants: Variants = {
         initial: (indices: { sourceIdx: number, targetIdx: number }) => ({ cx: adjustedGpus[indices.sourceIdx].x, cy: adjustedGpus[indices.sourceIdx].y, opacity: 0, scale: 0.5 }),
         animate: (indices: { sourceIdx: number, targetIdx: number }) => ({
             cx: adjustedGpus[indices.targetIdx].x,
             cy: adjustedGpus[indices.targetIdx].y,
             opacity: 1,
             scale: 1,
             transition: { duration: duration, delay: delay + indices.sourceIdx * 0.05 + indices.targetIdx * 0.02, ease: "easeInOut" }
         }),
         exit: { opacity: 0, scale: 0.5, transition: { duration: 0.2 } }
    };
    const lineVariants: Variants = { initial: { pathLength: 0, opacity: 0 }, animate: { pathLength: 1, opacity: 0.4, transition: { duration: duration*0.8, delay: delay } }, exit: { opacity: 0 } }; // Slightly faster line fade

    return (
        <AnimatePresence>
            {isActive && (
                <motion.svg className={styles.commSvgOverlay}>
                    {/* Lines */}
                    {adjustedGpus.map((source, i) => (
                        adjustedGpus.map((target, j) => ( // Line from each source to each target
                            <motion.path
                               key={`line-${i}-${j}`}
                               d={`M ${source.x} ${source.y} L ${target.x} ${target.y}`}
                               className={styles.commLine}
                               variants={lineVariants} initial="initial" animate="animate" exit="exit"
                           />
                        ))
                    ))}
                     {/* Packets: N*N packets moving */}
                     {adjustedGpus.map((_, sourceIdx) => (
                        adjustedGpus.map((_, targetIdx) => (
                             <motion.g key={`packet-group-${sourceIdx}-${targetIdx}`}>
                                 <motion.circle
                                     className={`${styles.dataPacketCircle} ${styles.weightData}`} // Often params
                                     custom={{ sourceIdx, targetIdx }}
                                     variants={packetVariants}
                                     initial="initial"
                                     animate="animate"
                                     exit="exit"
                                     r={8}
                                 />
                                 {/* Text inside packet */}
                                 <motion.text
                                      variants={{
                                           initial: { x: adjustedGpus[sourceIdx].x, y: adjustedGpus[sourceIdx].y + 3, opacity: 0 },
                                           animate: { x: adjustedGpus[targetIdx].x, y: adjustedGpus[targetIdx].y + 3, opacity: 1, transition: packetVariants.animate({ sourceIdx, targetIdx }).transition },
                                           exit: packetVariants.exit
                                      }}
                                       initial="initial" animate="animate" exit="exit"
                                       className={styles.packetText} textAnchor="middle"
                                  >
                                     {`${dataType[0]}${sourceIdx}`} {/* e.g., P0, P1 */}
                                  </motion.text>
                             </motion.g>
                        ))
                     ))}
                      {/* Central Label */}
                      {numGpus > 0 && <motion.text x="50%" y="40%" dominantBaseline="middle" textAnchor="middle" className={styles.commLabelLarge} initial={{opacity: 0}} animate={{opacity:1, transition:{delay: delay}}} exit={{opacity:0}}>AllGather ({dataType})</motion.text>}
                </motion.svg>
            )}
        </AnimatePresence>
    );
}; 