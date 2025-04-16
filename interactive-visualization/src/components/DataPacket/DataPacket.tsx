import React from 'react';
import styles from './DataPacket.module.css';

const DataPacket: React.FC = () => {
  // Simple visual representation of a micro-batch
  return (
    <div className={styles.dataPacket} title="Micro-batch">
      MB
    </div>
  );
};

export default DataPacket;
