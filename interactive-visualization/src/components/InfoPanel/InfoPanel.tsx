import React from 'react'; import styles from './InfoPanel.module.css';
const InfoPanel: React.FC = () => { return ( <div className={styles.infoPanel}> <p className={styles.note}>(Note: Activation memory visualization is simplified. Parameter/Gradient/OptState bars show persistent values unless otherwise indicated.)</p> </div> ); };
export default InfoPanel;
