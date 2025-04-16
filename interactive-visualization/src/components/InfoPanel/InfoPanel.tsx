import React from 'react';
import styles from './InfoPanel.module.css';
import { useSimulation } from '../../context/SimulationContext';

const InfoPanel: React.FC = () => {
  const { currentStep, stepDetails } = useSimulation();

  return (
    <div className={styles.infoPanel}>
      <p><strong>Step {currentStep}:</strong> {stepDetails?.description || 'Simulation Reset.'}</p>
      {/* Activation Memory Clarification */}
      <p className={styles.note}>(Note: Activation memory growth is simplified. Detailed visualization showing specific tensor storage for backprop will be added later.)</p>
    </div>
  );
};

export default InfoPanel;
