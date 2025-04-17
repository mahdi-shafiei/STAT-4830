import React from 'react';
import styles from './OperationDetailsPanel.module.css';
import { useSimulation } from '../../context/SimulationContext';
import MathDisplay from '../MathDisplay/MathDisplay'; // Ensure import

const OperationDetailsPanel: React.FC = () => {
  const { currentStep, stepDetails } = useSimulation();

  // Handle potential null stepDetails gracefully
  const description = stepDetails?.description || "Simulation Reset / Idle";
  const notation = stepDetails?.notation || null; // Pass null if no notation

  return (
    <div className={styles.detailsPanel}>
      <div className={styles.stepCounter}>Step {currentStep}</div>
      <div className={styles.description}>{description}</div>
      {/* Conditionally render notation section */}
      {notation && (
        <div className={styles.notation}>
          <span className={styles.eqLabel}>Eq:</span>
          {/* Ensure MathDisplay receives a string or null */}
          <MathDisplay texString={notation} />
        </div>
      )}
    </div>
  );
};

export default OperationDetailsPanel;
