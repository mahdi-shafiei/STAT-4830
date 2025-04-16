import React from 'react';
import styles from './ControlPanel.module.css';
import { useSimulation } from '../../context/SimulationContext'; // Import the hook

interface ControlPanelProps {
  // Prop to potentially update layout in App.tsx if needed, though context handles simulation count
  setDisplayNumGpus: (num: number) => void;
}

const ControlPanel: React.FC<ControlPanelProps> = ({ setDisplayNumGpus }) => {
  const {
      play, pause, nextStep, reset,
      setStrategy, strategy, // Get strategy state and setter
      setNumGpus, numGpus,   // Get GPU count state and setter from context
      isPlaying, currentStep, totalSteps
    } = useSimulation();

  const handleGpuChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const num = parseInt(event.target.value, 10);
    setNumGpus(num); // Update simulation GPU count in context
    setDisplayNumGpus(num); // Update layout display count in App
  };

  const handleStrategyChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const newStrategy = event.target.value;
    setStrategy(newStrategy); // Update strategy in context (will trigger reset)
    // Update display count based on new strategy if needed (e.g., single GPU)
    if (newStrategy === 'single') {
        setDisplayNumGpus(1);
    } else {
        // Optionally set a default for DP if switching from single
        // setDisplayNumGpus(4); // Or keep current numGpus from context
    }
  };

  // Determine if slider should be disabled (e.g., for 'single' strategy)
  const isSliderDisabled = strategy === 'single';

  return (
    <div className={styles.controlPanel}>
      <h2>Controls</h2>
      {/* Strategy Selector */}
      <div>
        <label htmlFor="strategy-select">Strategy: </label>
        <select id="strategy-select" value={strategy} onChange={handleStrategyChange}>
          <option value="single">Single GPU</option>
          <option value="dp">Data Parallel (DP)</option>
          {/* Add other strategies later */}
        </select>
      </div>

      {/* GPU Slider - Now enabled for DP */}
      <div>
        <label htmlFor="gpu-slider">Number of GPUs: {numGpus}</label> {/* Display context's GPU count */}
        <input
          type="range"
          id="gpu-slider"
          min="1"
          max="8" // Allow up to 8
          value={numGpus} // Control based on context state
          onChange={handleGpuChange}
          className={styles.slider}
          disabled={isSliderDisabled} // Disable only for 'single' strategy
        />
         {isSliderDisabled && <small>(Slider disabled for single-GPU mode)</small>}
      </div>

      {/* Simulation control buttons */}
      <div className={styles.simulationControls}>
        <button onClick={play} disabled={isPlaying || currentStep >= totalSteps}>Play</button>
        <button onClick={pause} disabled={!isPlaying}>Pause</button>
        <button onClick={nextStep} disabled={isPlaying || currentStep >= totalSteps}>Step Forward</button>
        <button onClick={reset}>Reset</button>
      </div>
       {/* Display current step */}
       <div>Step: {currentStep} / {totalSteps}</div>
    </div>
  );
};

export default ControlPanel;
