import React from 'react'; import styles from './InfoPanel.module.css'; import { useSimulation } from '../../context/SimulationContext';
// Remove MathDisplay import if no longer needed, or keep if used elsewhere
// import MathDisplay from '../MathDisplay/MathDisplay';

const InfoPanel: React.FC = () => { const { strategy } = useSimulation();
  const fsdpNote = "FSDP Note: Each layer's Params/Grads/OptStates ($w^{(k)}, \\hat{g}^{(k)}, Opt^{(k)}$) are sharded across GPUs. Params are temporarily AllGathered for computation.";
  const generalNote = "(Note: Activation memory is simplified but shows persistence until used in backprop).";

  return (
     <div className={styles.infoPanel}>
        <p className={styles.note}>
            {/* Render fsdpNote as plain text */} 
            {strategy === 'fsdp' ? fsdpNote : generalNote}
         </p>
         {/* Display generalNote separately if fsdpNote is shown */}
        {strategy === 'fsdp' && <p className={styles.note}>{generalNote}</p> }
     </div>
  );
 };
export default InfoPanel;
