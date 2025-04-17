import React from 'react';
import { motion, AnimatePresence, Variants } from 'framer-motion';
import styles from './CommAnimations.module.css';
import { Point } from '../../hooks/useGpuPositions';

interface AllReduceAnimProps { isActive: boolean; gpuPositions: Point[]; centerPos: Point; dataType?: string; containerOffset: Point; }

export const AllReduceAnim: React.FC<AllReduceAnimProps> = ({ isActive, gpuPositions, centerPos, dataType = 'Grad', containerOffset }) => {
     if (!isActive || gpuPositions.length === 0) return null;

     const adjustedCenter = { x: centerPos.x + containerOffset.x, y: centerPos.y + containerOffset.y };
     const adjustedGpus = gpuPositions.map(p => ({ x: p.x + containerOffset.x, y: p.y + containerOffset.y }));
     const duration = 0.4;
     const delayIn = 0.1;
     const delayOut = delayIn + duration + 0.2; // Time for packets to arrive + pause

    const packetVariantsIn: Variants = {
         initial: (i: number) => ({ cx: adjustedGpus[i].x, cy: adjustedGpus[i].y, opacity: 0, scale: 0.5 }),
         animate: { cx: adjustedCenter.x, cy: adjustedCenter.y, opacity: 1, scale: 1, transition: { duration: duration, delay: delayIn, ease: "easeIn" } },
         exit: { opacity: 0, scale: 0.5, transition: { duration: 0.1 } } // Fade at center
    };
    const packetVariantsOut: Variants = {
         initial: { cx: adjustedCenter.x, cy: adjustedCenter.y, opacity: 0, scale: 0.5 }, // Start at center hidden
         animate: (i: number) => ({ cx: adjustedGpus[i].x, cy: adjustedGpus[i].y, opacity: 1, scale: 1, transition: { duration: duration, delay: delayOut, ease: "easeOut" } }), // Move out after delay
         exit: { opacity: 0, scale: 0.5, transition: { duration: 0.2 } }
    };
     const lineVariantsIn: Variants = { initial: { pathLength: 0, opacity: 0 }, animate: { pathLength: 1, opacity: 0.6, transition: { duration: duration, delay: delayIn } }, exit: { opacity: 0 } };
     const lineVariantsOut: Variants = { initial: { pathLength: 0, opacity: 0 }, animate: { pathLength: 1, opacity: 0.6, transition: { duration: duration, delay: delayOut } }, exit: { opacity: 0 } };

    return (
        <AnimatePresence>
            {isActive && (
                <motion.svg className={styles.commSvgOverlay}>
                    {/* Lines In & Out */}
                    {adjustedGpus.map((gpuPos, i) => (
                        <React.Fragment key={`lines-${i}`}>
                            <motion.path d={`M ${gpuPos.x} ${gpuPos.y} L ${adjustedCenter.x} ${adjustedCenter.y}`} className={styles.commLine} variants={lineVariantsIn} initial="initial" animate="animate" exit="exit"/>
                            <motion.path d={`M ${adjustedCenter.x} ${adjustedCenter.y} L ${gpuPos.x} ${gpuPos.y}`} className={styles.commLine} variants={lineVariantsOut} initial="initial" animate="animate" exit="exit"/>
                        </React.Fragment>
                    ))}
                     {/* Packets In */}
                     {adjustedGpus.map((_, i) => (
                        <motion.circle key={`packet-in-${i}`} className={`${styles.dataPacketCircle} ${styles.genericData}`} custom={i} variants={packetVariantsIn} initial="initial" animate="animate" exit="exit" r={8} />
                     ))}
                     {/* Packets Out */}
                     {adjustedGpus.map((_, i) => (
                        <motion.g key={`packet-out-group-${i}`}>
                             <motion.circle className={`${styles.dataPacketCircle} ${styles.outputData}`} custom={i} variants={packetVariantsOut} initial="initial" animate="animate" exit="exit" r={8} />
                             <motion.text
                                 variants={{
                                      initial: { x: adjustedCenter.x, y: adjustedCenter.y + 3, opacity: 0 },
                                      animate: { x: adjustedGpus[i].x, y: adjustedGpus[i].y + 3, opacity: 1, transition: packetVariantsOut.animate(i).transition },
                                      exit: packetVariantsOut.exit
                                 }}
                                  initial="initial" animate="animate" exit="exit"
                                  className={styles.packetText} textAnchor="middle"
                             >Avg</motion.text>
                        </motion.g>
                     ))}
                      {/* Label */}
                      <motion.text x={adjustedCenter.x} y={adjustedCenter.y - 15} className={styles.commLabel} initial={{opacity: 0}} animate={{opacity:1, transition: {delay: delayIn}}} exit={{opacity:0}}>AllReduce ({dataType})</motion.text>
                </motion.svg>
            )}
        </AnimatePresence>
    );
}; 