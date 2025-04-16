import React from 'react';
import styles from './InfoPanel.module.css';
import { useSimulation } from '../../context/SimulationContext'; // Use context to check strategy

const InfoPanel: React.FC = () => {
  const { strategy } = useSimulation();

  const fsdpNote = "FSDP shards each layer's Params/Grads/OptStates across GPUs ($w^{(k)}, \\hat{g}^{(k)}, Opt^{(k)}$ on GPU $k$). Params are AllGathered temporarily for computation.";
  const generalNote = "(Note: Activation memory visualization is simplified.)";

  return (
    <div className={styles.infoPanel}>
      <p className={styles.note}>
          {strategy === 'fsdp' ? fsdpNote : generalNote}
          {strategy !== 'fsdp' && <br/>} {/* Add break if only general note shown */}
          {strategy === 'fsdp' && generalNote} {/* Always show activation note */}
      </p>
    </div>
  );
};
export default InfoPanel;
