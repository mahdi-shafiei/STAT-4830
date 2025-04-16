import React, { useState } from 'react';
import styles from './App.module.css';
import ControlPanel from './components/ControlPanel/ControlPanel';
import VisualizationArea from './components/VisualizationArea/VisualizationArea';
import InfoPanel from './components/InfoPanel/InfoPanel';
import { SimulationProvider } from './context/SimulationContext';

function App() {
  // State to control the layout (# GPUs displayed)
  const [displayNumGpus, setDisplayNumGpus] = useState(4); // Default to 4 for DP demo

  return (
    // Pass initial GPU count to context provider
    <SimulationProvider initialNumGpus={displayNumGpus}>
      <div className={styles.appContainer}>
        <header className={styles.header}>
          {/* Update Title for Chunk 3 */}
          <h1>Transformer Parallelism Visualization (Chunk 3 - DP)</h1>
        </header>
        <div className={styles.mainContent}>
          <aside className={styles.controlPanel}>
            {/* Control panel now reads simulation state from context */}
            {/* Pass setter for potentially updating layout if needed externally */}
            <ControlPanel setDisplayNumGpus={setDisplayNumGpus} />
          </aside>
          <main className={styles.visualizationArea}>
            {/* VisualizationArea reads numGpus from context */}
            <VisualizationArea />
          </main>
        </div>
        <footer className={styles.infoPanel}>
          <InfoPanel />
        </footer>
      </div>
    </SimulationProvider>
  );
}

export default App;
