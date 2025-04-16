import React from 'react';
import { motion } from 'framer-motion';
import styles from './CommunicationArrow.module.css';

// Define specific operation and data types for better type safety
export type CommOperation = 'AllReduce' | 'P2P' | 'AllGather' | 'ReduceScatter' | 'AlltoAll';
export type CommDataType = 'Activations' | 'Gradients' | 'Params' | 'Tokens' | 'KV';

interface CommunicationArrowProps {
  type: CommOperation;
  dataType: CommDataType;
  isActive: boolean; // Controls visibility/animation state
}

const CommunicationArrow: React.FC<CommunicationArrowProps> = ({ type, dataType, isActive }) => {
  if (!isActive) {
    return null; // Don't render if not active
  }

  const variants = {
    hidden: { opacity: 0, scale: 0.8, y: 10 }, // Start slightly below and scaled down
    visible: { opacity: 1, scale: 1, y: 0, transition: { duration: 0.3, ease: "easeOut" } },
    exit: { opacity: 0, scale: 0.8, y: -10, transition: { duration: 0.2, ease: "easeIn" } } // Exit upwards
  };

  // Simple visual representation centered on the screen
  return (
    <motion.div
        className={styles.communicationOverlay}
        variants={variants}
        initial="hidden"
        animate="visible"
        exit="exit"
    >
        {/* Use dataType to potentially style the box border/background */}
        <div className={`${styles.arrowBox} ${styles[dataType.toLowerCase()]}`}>
            <span className={styles.arrowLabel}>{type}</span>
            <span className={styles.dataTypeLabel}>({dataType})</span>
            <div className={styles.flowIndicator}>➔➔➔</div>
        </div>
    </motion.div>
  );
};

export default CommunicationArrow;
