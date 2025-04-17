import React from 'react';
import { motion, AnimatePresence, Variants } from 'framer-motion';
import styles from './CommAnimations.module.css';
import { Point } from '../../hooks/useGpuPositions';

interface AllReduceAnimProps { isActive: boolean; gpuPositions: Point[]; centerPos: Point; dataType?: string | null; containerOffset?: Point; }

export const AllReduceAnim: React.FC<AllReduceAnimProps> = ({ isActive, gpuPositions, centerPos, dataType, containerOffset = {x:0, y:0} }) => {
     if (!isActive || gpuPositions.length === 0) return null;

     const adjustedCenter = { x: centerPos.x + containerOffset.x, y: centerPos.y + containerOffset.y };
     const adjustedGpus = gpuPositions.map(p => ({ x: p.x + containerOffset.x, y: p.y + containerOffset.y }));
     const duration = 0.4;
     const delayIn = 0.1;
     const delayOut = delayIn + duration + 0.3; // Short pause at center
     const baseDataType = dataType || 'Data';

    const packetVariantsIn: Variants = {
         initial: (i: number) => ({ cx: adjustedGpus[i].x, cy: adjustedGpus[i].y, opacity: 0, scale: 0.5 }),
         animate: { cx: adjustedCenter.x, cy: adjustedCenter.y, opacity: 1, scale: 1, transition: { duration: duration, delay: delayIn, ease: "easeIn" } },
         exit: { opacity: 0, scale: 0.1, transition: { duration: 0.1 } } // Quick shrink at center
    };
    const textVariantsIn: Variants = {
         initial: (i: number) => ({ x: adjustedGpus[i].x, y: adjustedGpus[i].y + 3, opacity: 0 }),
         animate: { x: adjustedCenter.x, y: adjustedCenter.y + 3, opacity: 1, transition: { duration: duration, delay: delayIn, ease: "easeIn" } },
         exit: { opacity: 0 }
    }
    const packetVariantsOut: Variants = {
         initial: { cx: adjustedCenter.x, cy: adjustedCenter.y, opacity: 0, scale: 0.5 },
         animate: (i: number) => ({ cx: adjustedGpus[i].x, cy: adjustedGpus[i].y, opacity: 1, scale: 1, transition: { duration: duration, delay: delayOut, ease: "easeOut" } }),
         exit: { opacity: 0, scale: 0.5, transition: { duration: 0.2 } }
    };
     const textVariantsOut: Variants = {
         initial: { x: adjustedCenter.x, y: adjustedCenter.y + 3, opacity: 0 },
         animate: (i: number) => ({ x: adjustedGpus[i].x, y: adjustedGpus[i].y + 3, opacity: 1, transition: { duration: duration, delay: delayOut, ease: "easeOut" } }),
         exit: { opacity: 0 }
     }
     const lineVariants: Variants = { initial: { pathLength: 0, opacity: 0 }, animate: (isIn: boolean) => ({ pathLength: 1, opacity: 0.5, transition: { duration: duration*0.8, delay: isIn ? delayIn : delayOut } }), exit: { opacity: 0 } };
     const labelText = `AllReduce${dataType ? ` (${dataType})` : ''}`;

    return (
        <AnimatePresence>
            {isActive && (
                <motion.svg className={styles.commSvgOverlay} key="allreduce-svg">
                    {/* Lines In & Out */}
                    {adjustedGpus.map((gpuPos, i) => (
                        <React.Fragment key={`lines-${i}`}>
                            <motion.path d={`M ${gpuPos.x} ${gpuPos.y} L ${adjustedCenter.x} ${adjustedCenter.y}`} className={styles.commLine} custom={true} variants={lineVariants} initial="initial" animate="animate" exit="exit"/>
                            <motion.path d={`M ${adjustedCenter.x} ${adjustedCenter.y} L ${gpuPos.x} ${gpuPos.y}`} className={styles.commLine} custom={false} variants={lineVariants} initial="initial" animate="animate" exit="exit"/>
                        </React.Fragment>
                    ))}
                     {/* Packets In */}
                     {adjustedGpus.map((_, i) => (
                        <motion.g key={`packet-in-group-${i}`}>
                            <motion.circle className={`${styles.dataPacketCircle} ${styles.genericData}`} custom={i} variants={packetVariantsIn} initial="initial" animate="animate" exit="exit" r={10} />
                            <motion.text custom={i} variants={textVariantsIn} initial="initial" animate="animate" exit="exit" className={styles.packetText} textAnchor="middle">
                               {`${baseDataType[0]}${i}`} {/* e.g., G0, G1 */}
                            </motion.text>
                        </motion.g>
                     ))}
                     {/* Packets Out */}
                     {adjustedGpus.map((_, i) => (
                        <motion.g key={`packet-out-group-${i}`}>
                             <motion.circle className={`${styles.dataPacketCircle} ${styles.outputData}`} custom={i} variants={packetVariantsOut} initial="initial" animate="animate" exit="exit" r={10} />
                             <motion.text custom={i} variants={textVariantsOut} initial="initial" animate="animate" exit="exit" className={styles.packetText} textAnchor="middle">
                                Avg {/* Indicate averaged result */}
                             </motion.text>
                        </motion.g>
                     ))}
                      {/* Label */}
                      <motion.text x={adjustedCenter.x} y={adjustedCenter.y - 15} className={styles.commLabelLarge} initial={{opacity: 0}} animate={{opacity:1, transition: {delay: delayIn}}} exit={{opacity:0}}>{labelText}</motion.text>
                </motion.svg>
            )}
        </AnimatePresence>
    );
}; 