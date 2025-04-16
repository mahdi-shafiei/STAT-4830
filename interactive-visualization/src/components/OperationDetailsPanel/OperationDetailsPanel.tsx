import React from 'react';
import styles from './OperationDetailsPanel.module.css';
import { useSimulation } from '../../context/SimulationContext';
import MathDisplay from '../MathDisplay/MathDisplay'; // Import the new component
import { AnimatePresence, motion } from 'framer-motion';

const OperationDetailsPanel: React.FC = () => {
  const { currentStep, stepDetails } = useSimulation();

  const description = stepDetails?.description || "Simulation Idle.";
  const notation = stepDetails?.notation; // Get notation directly

  // Animation variants
  const panelVariants = {
    hidden: { opacity: 0, y: -10 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.3 } },
    exit: { opacity: 0, y: 10, transition: { duration: 0.2 } }
  };

  return (
    <AnimatePresence mode="wait">
      <motion.div
          key={currentStep} // Change key forces animation on step change
          className={styles.detailsPanel}
          variants={panelVariants}
          initial="hidden"
          animate="visible"
          exit="exit"
       >
        <div className={styles.stepCounter}>Step {currentStep}</div>
        <div className={styles.description}>{description}</div>
        {notation && (
          <div className={styles.notation}>
            <span className={styles.notationLabel}>Eq:</span>
            <MathDisplay texString={notation} />
          </div>
        )}
      </motion.div>
    </AnimatePresence>
  );
};

export default OperationDetailsPanel;
