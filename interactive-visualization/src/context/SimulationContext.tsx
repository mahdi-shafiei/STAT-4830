import React, { createContext, useState, useContext, useEffect, useRef, useCallback } from 'react';
import { DP_STEPS, DP_TOTAL_STEPS } from '../config/strategies/dp';

// --- Type Definitions ---
// Define structure for a single GPU's state
export interface GpuState {
    id: number;
    paramMemory: number;
    activationMemory: number;
    gradientMemory: number;
    optStateMemory: number;
    status: 'idle' | 'computing' | 'communicating';
    currentLayerName?: string; // Layer being processed
}

// Define communication operation and data types explicitly
export type CommOperation = 'AllReduce' | 'P2P' | 'AllGather' | 'ReduceScatter' | 'AlltoAll' | string; // Allow string for flexibility
export type CommDataType = 'Activations' | 'Gradients' | 'Params' | 'Tokens' | 'KV' | string;

// Define StepDetail structure (can be expanded)
export interface StepDetail {
    step: number;
    type?: 'INIT' | 'COMPUTE' | 'COMM' | 'GRADIENTS' | 'UPDATE' | 'DONE' | string; // Allow custom types
    layer?: string;
    parallel?: boolean; // True if step happens across multiple ranks simultaneously
    direction?: 'forward' | 'backward';
    operation?: CommOperation;
    dataType?: CommDataType;
    description: string;
}

// --- Simulation Constants ---
const MAX_ACTIVATION = 75;
const MAX_PARAM = 80;
const MAX_OPTSTATE = 80;
const MAX_GRADIENT = 50; // Placeholder max gradient size per GPU before reduction

// --- Helper Functions ---
// Function to generate steps for single GPU (similar to Chunk 2)
const generateSingleGpuSteps = (): StepDetail[] => {
    const MODEL_LAYERS = ['Embed', 'MHA', 'FFN', 'LN', 'Output'];
    const steps: StepDetail[] = [{ step: 0, type: 'INIT', description: 'Single GPU Initial state.' }];
    MODEL_LAYERS.forEach((layer, index) => {
        steps.push({
            step: index + 1, type: 'COMPUTE', direction: 'forward', layer: layer,
            description: `Forward Pass: Processing ${layer} layer.`
        });
    });
    steps.push({ step: MODEL_LAYERS.length + 1, type: 'DONE', description: 'Forward Pass Complete.' });
    // Add backward/update steps later
    return steps.map((s, index) => ({ ...s, step: index })); // Renumber steps
};
const SINGLE_GPU_STEPS = generateSingleGpuSteps();
const SINGLE_GPU_TOTAL_STEPS = SINGLE_GPU_STEPS.length > 0 ? SINGLE_GPU_STEPS.length - 1 : 0;


// Function to initialize GPU states based on strategy and number
const initializeGpuStates = (numGpus: number, strategy: string): GpuState[] => {
    console.log(`Initializing state for ${numGpus} GPUs, strategy: ${strategy}`);
    return Array.from({ length: numGpus }, (_, i) => {
        // Determine initial memory based on strategy
        // DP replicates Params and OptStates fully
        const initialParams = (strategy === 'dp' || strategy === 'single') ? MAX_PARAM : 0;
        const initialOptStates = (strategy === 'dp' || strategy === 'single') ? MAX_OPTSTATE : 0;

        return {
            id: i,
            paramMemory: initialParams,
            activationMemory: 0,
            gradientMemory: 0,
            optStateMemory: initialOptStates,
            status: 'idle',
            currentLayerName: undefined,
        };
    });
};

// --- Context Definition ---
interface SimulationState {
  currentStep: number;
  totalSteps: number;
  isPlaying: boolean;
  strategy: string;
  numGpus: number; // Number of GPUs currently simulated
  gpuStates: GpuState[]; // State of each GPU
  stepDetails: StepDetail | null; // Details of the current step
}

interface SimulationContextProps extends SimulationState {
  play: () => void;
  pause: () => void;
  nextStep: () => void;
  reset: () => void;
  setStrategy: (strategy: string) => void;
  setNumGpus: (num: number) => void;
}

const SimulationContext = createContext<SimulationContextProps | undefined>(undefined);

// --- Context Provider ---
export const SimulationProvider: React.FC<{ children: React.ReactNode; initialNumGpus?: number }> = ({ children, initialNumGpus = 1 }) => {
  const [strategy, setStrategyInternal] = useState('single'); // Default strategy
  const [numGpus, setNumGpusInternal] = useState(initialNumGpus);
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [gpuStates, setGpuStates] = useState<GpuState[]>(() => initializeGpuStates(numGpus, strategy));

  const intervalRef = useRef<number | null>(null);

  // Determine current simulation steps and total based on strategy
  const { currentSimulationSteps, totalSteps } = React.useMemo(() => {
    if (strategy === 'dp') {
        return { currentSimulationSteps: DP_STEPS, totalSteps: DP_TOTAL_STEPS };
    }
    // Default to single GPU
    return { currentSimulationSteps: SINGLE_GPU_STEPS, totalSteps: SINGLE_GPU_TOTAL_STEPS };
  }, [strategy]);

  // Get details for the current step
  const stepDetails = currentSimulationSteps[currentStep] || null;


  // --- State Update Logic (including DP) ---
  const updateGpuStatesForStep = useCallback((step: number) => {
    const details = currentSimulationSteps[step];
    if (!details) return; // Exit if step details not found

    setGpuStates(prevStates => {
        // Determine if the step applies to all GPUs (parallel) or just one (for future strategies)
        const isParallelStep = details.parallel === true || strategy === 'single'; // Single GPU steps apply to the only GPU

        return prevStates.map((gpu, index) => {
            // Only update GPUs relevant to the step (all for parallel, specific index otherwise)
            if (!isParallelStep /* && index !== targetGpuIndex */) { // Add targetGpuIndex later if needed
                return gpu; // Return unchanged state if step doesn't apply
            }

            // Copy previous state for modification
            let newState = { ...gpu };
            let nextStatus: GpuState['status'] = 'idle'; // Default next status
            newState.currentLayerName = undefined; // Clear layer name by default

            // --- Logic based on step type ---
            switch (details.type) {
                case 'INIT':
                    // Handled by initializeGpuStates on reset/strategy change
                    nextStatus = 'idle';
                    newState.activationMemory = 0;
                    newState.gradientMemory = 0;
                    break;
                case 'COMPUTE':
                    nextStatus = 'computing';
                    newState.currentLayerName = details.layer;
                    if (details.direction === 'forward') {
                        // Simplified activation growth (spread over forward steps)
                        const forwardSteps = currentSimulationSteps.filter(s => s.type === 'COMPUTE' && s.direction === 'forward').length;
                        const growthPerStep = forwardSteps > 0 ? MAX_ACTIVATION / forwardSteps : MAX_ACTIVATION;
                        newState.activationMemory = Math.min(gpu.activationMemory + growthPerStep, MAX_ACTIVATION);
                    }
                    // Add backward compute logic later
                    break;
                case 'GRADIENTS':
                    // Placeholder: Max out gradient memory locally *before* communication
                    nextStatus = 'computing'; // Still active computing gradients
                    newState.currentLayerName = 'Gradients';
                    newState.gradientMemory = MAX_GRADIENT;
                    newState.activationMemory = 0; // Clear activations (simplified)
                    break;
                case 'COMM':
                    nextStatus = 'communicating';
                    // Visual state change handled here. Memory changes usually reflect *after* comms.
                    break;
                case 'UPDATE':
                    nextStatus = 'computing'; // Treat update as compute activity
                    newState.currentLayerName = 'Optimizer';
                     // Gradients are consumed by the optimizer (reset after this step)
                    break;
                 case 'DONE':
                    nextStatus = 'idle';
                    break;
                default:
                    nextStatus = 'idle'; // Default for unknown types
                    break;
            }

            // --- Handle state changes *after* specific steps ---
            // Find the *previous* step's details to react to completed actions
            const prevStepDetails = currentSimulationSteps[step - 1];
            if (prevStepDetails) {
                if (prevStepDetails.type === 'COMM' && prevStepDetails.operation === 'AllReduce' && prevStepDetails.dataType === 'Gradients') {
                    // After AllReduce, gradients are conceptually averaged
                    newState.gradientMemory = MAX_GRADIENT * 0.6; // Visual: show reduced gradients
                } else if (prevStepDetails.type === 'UPDATE') {
                     // After optimizer update, gradients are conceptually zeroed
                     newState.gradientMemory = 0;
                }
            }
             newState.status = nextStatus; // Set the final status for this GPU for this step
            return newState;
        });
    });
  }, [currentSimulationSteps, strategy]); // Dependencies

  // --- Simulation Controls (Play, Pause, Step, Reset) ---
  const clearSimulationInterval = useCallback(() => {
    if (intervalRef.current !== null) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  const advanceStep = useCallback(() => {
    setCurrentStep(prev => {
      const next = prev + 1;
      if (next > totalSteps) {
        setIsPlaying(false);
        clearSimulationInterval();
        updateGpuStatesForStep(totalSteps); // Ensure final state update
        return totalSteps;
      }
      updateGpuStatesForStep(next); // Update GPU states for the NEW step
      return next;
    });
  }, [totalSteps, clearSimulationInterval, updateGpuStatesForStep]);

  const play = useCallback(() => {
    if (isPlaying || currentStep >= totalSteps) return;
    setIsPlaying(true);
    clearSimulationInterval();
    advanceStep(); // Advance first step immediately

    if (currentStep + 1 <= totalSteps) { // Ensure interval runs if steps remain
        intervalRef.current = window.setInterval(advanceStep, 1200); // Speed
    } else {
         setIsPlaying(false); // Reached end after first advance
    }
  }, [isPlaying, currentStep, totalSteps, clearSimulationInterval, advanceStep]);

  const pause = useCallback(() => {
    if (!isPlaying) return;
    setIsPlaying(false);
    clearSimulationInterval();
  }, [isPlaying, clearSimulationInterval]);

  // Reset function now considers strategy and gpu count
  const reset = useCallback((currentNumGpus = numGpus, currentStrategy = strategy) => {
    console.log(`Reset called. numGpus: ${currentNumGpus}, strategy: ${currentStrategy}`);
    setIsPlaying(false);
    clearSimulationInterval();
    setCurrentStep(0);
    setGpuStates(initializeGpuStates(currentNumGpus, currentStrategy));
    // updateGpuStatesForStep(0); // Call update for step 0 state
  }, [numGpus, strategy, clearSimulationInterval]); // Include numGpus and strategy

  // --- Strategy and GPU Count Management ---
  const setNumGpusCallback = useCallback((num: number) => {
      // Avoid resetting if count hasn't changed
      if (num !== numGpus) {
        console.log("Setting num GPUs:", num);
        setNumGpusInternal(num);
        reset(num, strategy); // Reset simulation with new GPU count
      }
  }, [numGpus, strategy, reset]); // Correct dependencies

  const setStrategyCallback = useCallback((newStrategy: string) => {
      // Avoid resetting if strategy hasn't changed
      if (newStrategy !== strategy) {
        console.log("Setting strategy:", newStrategy);
        const nextNumGpus = newStrategy === 'single' ? 1 : numGpus; // Force 1 GPU for single mode
        setStrategyInternal(newStrategy);
        setNumGpusInternal(nextNumGpus); // Update internal count
        reset(nextNumGpus, newStrategy); // Reset with new strategy and count
      }
  }, [numGpus, strategy, reset]); // Correct dependencies

  // Effect to initialize state correctly on first load or when strategy/gpu changes
   useEffect(() => {
     console.log("Strategy or numGpus changed, resetting state for step 0");
     // This reset ensures gpuStates aligns with numGpus and strategy
     reset(numGpus, strategy);
   // eslint-disable-next-line react-hooks/exhaustive-deps
   }, [strategy, numGpus]); // Run reset when these external controls change state


  // Cleanup interval on component unmount
  useEffect(() => clearSimulationInterval, [clearSimulationInterval]);

  // Context value provided to consumers
  const value = {
    currentStep,
    totalSteps,
    isPlaying,
    strategy,
    numGpus, // Provide actual number used
    gpuStates,
    stepDetails,
    play,
    pause,
    nextStep: advanceStep,
    reset: () => reset(), // Ensure reset uses current state values
    setStrategy: setStrategyCallback,
    setNumGpus: setNumGpusCallback,
  };

  return (
    <SimulationContext.Provider value={value}>
      {children}
    </SimulationContext.Provider>
  );
};

// Custom hook to use the simulation context
export const useSimulation = () => {
  const context = useContext(SimulationContext);
  if (context === undefined) {
    throw new Error('useSimulation must be used within a SimulationProvider');
  }
  return context;
};

// Note: generateForwardSteps is internal now and not exported
