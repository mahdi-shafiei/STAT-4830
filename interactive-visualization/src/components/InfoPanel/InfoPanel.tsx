import React from 'react'; import styles from './InfoPanel.module.css'; import { useSimulation } from '../../context/SimulationContext'; import MathDisplay from '../MathDisplay/MathDisplay';
const InfoPanel: React.FC = () => { const { strategy } = useSimulation();
  const fsdpNote = "FSDP Note: Each layer's Params/Grads/OptStates ($w^{(k)}, \\hat{g}^{(k)}, Opt^{(k)}$) are sharded across GPUs. Params are temporarily AllGathered for computation.";
  const generalNote = "(Note: Activation memory is simplified but shows persistence until used in backprop).";
  return ( <div className={styles.infoPanel}> <p className={styles.note}> {strategy === 'fsdp' ? <MathDisplay texString={fsdpNote} /> : generalNote} </p> {strategy === 'fsdp' && <p className={styles.note}>{generalNote}</p> } </div> ); };
export default InfoPanel;
