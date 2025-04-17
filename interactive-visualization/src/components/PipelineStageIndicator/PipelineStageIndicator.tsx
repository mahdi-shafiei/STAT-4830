import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import styles from './PipelineStageIndicator.module.css';

interface PipelineStageIndicatorProps {
  layers: string[];
  currentLayer: string | undefined | null;
}

const PipelineStageIndicator: React.FC<PipelineStageIndicatorProps> = ({ layers, currentLayer }) => {
    const variants = {
        inactive: { backgroundColor: "#e9ecef", borderColor: "#ced4da", color: "#6c757d" },
        active: { backgroundColor: "#d1e7dd", borderColor: "#a3cfbb", color: "#0f5132", scale: 1.05, transition: { duration: 0.2} },
    };

    return (
        <div className={styles.pipelineIndicatorContainer}>
            {layers.map((layer) => (
                <motion.div
                    key={layer}
                    className={styles.layerBox}
                    variants={variants}
                    animate={currentLayer === layer ? "active" : "inactive"}
                    initial="inactive" // Prevent animation on initial load
                    layout // Smoothly animate layout changes if container resizes
                >
                    {layer}
                </motion.div>
            ))}
        </div>
    );
};

export default PipelineStageIndicator; 