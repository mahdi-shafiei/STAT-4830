import React, { createContext, useState, useContext, useEffect, useRef, useCallback } from 'react';
import { generateDpSteps } from '../config/strategies/dp';
import { generateFsdpSteps } from '../config/strategies/fsdp';
import { generateSingleGpuSteps } from './singleGpuSteps';
import { generateTpSteps } from '../config/strategies/tp';
import type { GpuState, CommOperation, CommDataType, StepDetail } from './types';

export type { GpuState, CommOperation, CommDataType, StepDetail };

const MAX_ACTIVATION = 100; const MAX_PARAM = 100; const MAX_OPTSTATE = 100; const MAX_GRADIENT = 100;
const MODEL_LAYERS = ['Embed', 'MHA', 'FFN', 'LN', 'Output']; // Must match layers used in step generators

// --- Helper: Initialize GPU States ---
const initializeGpuStates = (numGpus: number, strategy: string): GpuState[] => {
    const count = Math.max(1, numGpus); console.log(`Initializing state for ${count} GPUs, strategy: ${strategy}`);
    return Array.from({ length: count }, (_, i) => {
        let iP = 0, iO = 0, iG = 0; const shardDenom = (strategy === 'fsdp' && count > 0) ? count : 1; const isParallel = strategy === 'dp' || strategy === 'fsdp' || strategy === 'tp';
        if (strategy === 'single' || strategy === 'dp') { iP = MAX_PARAM; iO = MAX_OPTSTATE; } else if (strategy === 'fsdp') { iP = MAX_PARAM / shardDenom; iO = MAX_OPTSTATE / shardDenom; } else if (strategy === 'tp') {
            // TP: Only some layers sharded. Assume ~50% sharded for visualization placeholder.
            // More accurate would be to sum sizes based on layer types.
            const shardedFraction = 0.5; // Estimate
            const fullFraction = 1.0 - shardedFraction;
            iP = (MAX_PARAM * shardedFraction) / 2 + (MAX_PARAM * fullFraction);
            iO = (MAX_OPTSTATE * shardedFraction) / 2 + (MAX_OPTSTATE * fullFraction);
            // Gradients start at 0
        }
        return { id: i, paramMemory: iP, activationMemory: 0, gradientMemory: iG, optStateMemory: iO, status: 'idle', currentLayerName: undefined, isParamsTempFull: false, dataShardId: isParallel ? i + 1 : undefined, };
    });
};

// --- Helper: Pre-calculate Activation Memory Profile (REFINED) ---
const calculateActivationProfile = (steps: StepDetail[]): number[] => {
    const profile: number[] = Array(steps.length).fill(0);
    if (steps.length === 0) return profile; // Handle empty steps case

    const activationMemoryCosts: Record<string, number> = {};
    const activationReleaseStep: Record<string, number> = {};

    // Assign memory cost per layer (can be refined later)
    const costPerLayer = MAX_ACTIVATION / MODEL_LAYERS.length;
    MODEL_LAYERS.forEach(layer => activationMemoryCosts[layer] = costPerLayer);
    activationMemoryCosts['Input'] = 0; // No cost for input

    // Find the step *immediately after* an activation is last needed
    steps.forEach((step, index) => {
        // Layer L's backward compute consumes activation from L-1
        if (step.type === 'COMPUTE' && step.direction === 'backward' && step.activationConsumedLayer) {
            const releaseAfterStepIndex = index; // Activation needed *during* this step
             // We only update if this release step is *earlier* than a previously recorded one
            if (!activationReleaseStep[step.activationConsumedLayer] || activationReleaseStep[step.activationConsumedLayer] > releaseAfterStepIndex + 1) {
                 activationReleaseStep[step.activationConsumedLayer] = releaseAfterStepIndex + 1; // Memory freed *after* this step completes
            }
        }
        // DP: Simplified GRADIENTS step consumes all activations (handled by DONE)
        // DONE step releases anything still held
        if (step.type === 'DONE') {
             MODEL_LAYERS.forEach(layer => {
                if (!activationReleaseStep[layer]) {
                    activationReleaseStep[layer] = index; // Release during DONE step (or slightly before)
                }
             });
             // Also release 'Input' if somehow held
             if (!activationReleaseStep['Input']) {
                  activationReleaseStep['Input'] = index;
             }
        }
    });
    // Ensure all layers have a release point (default to DONE step if not consumed earlier)
    const doneStepIndex = steps.findIndex(s => s.type === 'DONE');
    if (doneStepIndex !== -1) {
      MODEL_LAYERS.forEach(layer => {
        if (!activationReleaseStep[layer]) {
           activationReleaseStep[layer] = doneStepIndex; // Default release at DONE
        }
      });
       if (!activationReleaseStep['Input']) {
            activationReleaseStep['Input'] = doneStepIndex;
       }
    }
    console.log("Activation Release Step Indices (Memory Freed After Step i):", activationReleaseStep);

    // Calculate cumulative memory profile based on production and release steps
    let memoryHeld = 0;
    const activeActivations = new Set<string>();

    for (let i = 0; i < steps.length; i++) {
        const stepDetails = steps[i];

        // Check for releases *before* processing the current step's production
        const releasedLayers: string[] = [];
        activeActivations.forEach(layerName => {
            if (activationReleaseStep[layerName] === i) {
                memoryHeld -= (activationMemoryCosts[layerName] ?? 0);
                releasedLayers.push(layerName);
                 if (steps[0]?.strategy === 'dp') { // Log only for DP
                     console.log(`DP Profile Calc Step ${i}: Releasing activation for ${layerName}, Mem Held: ${memoryHeld}`);
                 }
            }
        });
        releasedLayers.forEach(layerName => activeActivations.delete(layerName));

        // Add memory for activations produced by the current step
        let producedLayerName: string | null = null;
        if (stepDetails?.activationProduced) { // Check the explicit field
            producedLayerName = stepDetails.activationProduced;
        }

        if (producedLayerName && !activeActivations.has(producedLayerName)) {
             activeActivations.add(producedLayerName);
             memoryHeld += (activationMemoryCosts[producedLayerName] ?? 0);
             if (steps[0]?.strategy === 'dp') { // Log only for DP
                 console.log(`DP Profile Calc Step ${i}: Producing activation for ${producedLayerName}, Mem Held: ${memoryHeld}`);
             }
        }

        // Assign the calculated memory level for this step
        profile[i] = Math.max(0, Math.min(memoryHeld, MAX_ACTIVATION)); // Clamp memory

        // Ensure memory is zero at INIT and DONE steps explicitly
        if (stepDetails?.type === 'INIT') profile[i] = 0;
        // Don't force zero at DONE, let release logic handle it
        // if (stepDetails?.type === 'DONE') profile[i] = 0;
    }
    // Ensure final step (DONE) has zero activation memory
    if (profile.length > 0 && steps[profile.length - 1]?.type === 'DONE') {
       profile[profile.length - 1] = 0;
    }

    // Log the final profile only for DP
    if (steps[0]?.strategy === 'dp') {
        console.log("DP Calculated Activation Memory Profile:", profile);
    }
    // console.log("Calculated Activation Memory Profile:", profile);
    return profile;
};


// --- Context Definition ---
interface SimulationState { currentStep: number; totalSteps: number; isPlaying: boolean; strategy: string; numGpus: number; gpuStates: GpuState[]; stepDetails: StepDetail | null; }
interface SimulationContextProps extends SimulationState { play: () => void; pause: () => void; nextStep: () => void; prevStep: () => void; reset: () => void; setStrategy: (strategy: string) => void; setNumGpus: (num: number) => void; }
const SimulationContext = createContext<SimulationContextProps | undefined>(undefined);

// --- Context Provider ---
export const SimulationProvider: React.FC<{ children: React.ReactNode; initialNumGpus?: number }> = ({ children, initialNumGpus = 1 }) => {
  const initialStrategy = initialNumGpus > 1 ? 'fsdp' : 'single';
  const validatedInitialNumGpus = initialStrategy === 'single' ? 1 : Math.max(2, initialNumGpus);
  const [strategy, setStrategyInternal] = useState(initialStrategy);
  const [numGpus, setNumGpusInternal] = useState(validatedInitialNumGpus);
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [gpuStates, setGpuStates] = useState<GpuState[]>(() => initializeGpuStates(numGpus, strategy));
  const intervalRef = useRef<number | null>(null);

  // Generate steps AND activation profile - UPDATED FOR TP
  const { currentSimulationSteps, totalSteps, activationMemoryProfile } = React.useMemo(() => {
    let steps: StepDetail[];
    if (strategy === 'dp') steps = generateDpSteps(numGpus);
    else if (strategy === 'fsdp') steps = generateFsdpSteps(numGpus);
    else if (strategy === 'tp') steps = generateTpSteps();
    else steps = generateSingleGpuSteps();
    const profile = calculateActivationProfile(steps);
    const totSteps = steps.length > 0 ? steps.length - 1 : 0;
    console.log(`Generated ${steps.length} steps for strategy ${strategy} with ${numGpus} GPUs.`);
    return { currentSimulationSteps: steps, totalSteps: totSteps, activationMemoryProfile: profile };
}, [strategy, numGpus]);

  const stepDetails = currentSimulationSteps[currentStep] || null;

  // --- State Update Logic (Apply Activation Profile, Fix DP Grad Clear) --- UPDATED FOR TP
  const updateGpuStatesForStep = useCallback((step: number) => {
    if (!currentSimulationSteps || currentSimulationSteps.length === 0) return;
    const clampedStep = Math.max(0, Math.min(step, totalSteps));
    const details = currentSimulationSteps[clampedStep];
    if (!details) { console.warn(`No step details for step ${clampedStep}`); return; }

    // Get pre-calculated activation memory for this step
    const calculatedActivationMemory = activationMemoryProfile[clampedStep] ?? 0;

    setGpuStates(prevStates => {
        // Ensure prevStates are valid before mapping
        const expectedGpuCount = strategy === 'single' ? 1 : (strategy === 'tp' ? 2 : numGpus);
        if (!Array.isArray(prevStates) || prevStates.length !== expectedGpuCount) {
             console.warn(`GPU state mismatch or invalid. Expected ${expectedGpuCount}, got ${prevStates?.length}. Reinitializing.`);
             return initializeGpuStates(expectedGpuCount, strategy);
        }

        const isParallel = details.parallel === true || strategy === 'tp'; // TP is always parallel (N=2)
        const isParamsSharded = strategy === 'fsdp' || strategy === 'tp'; // TP also shards some params
        const isGradsSharded = strategy === 'fsdp'; // Simplify: Only FSDP explicitly shards grads this way in viz
        const isOptStatesSharded = strategy === 'fsdp' || strategy === 'tp'; // TP shards some opt states
        const shardDenom = strategy === 'fsdp' ? numGpus : (strategy === 'tp' ? 2 : 1); // N_tp = 2 for TP
        const prevStepDetails = (clampedStep > 0) ? currentSimulationSteps[clampedStep - 1] : null;

        return prevStates.map((gpu) => {
            // TP always applies to all GPUs in the group (N=2)
            // if (!isParallel && strategy !== 'single' && gpu.id !== 0) return gpu; // Remove this line? TP applies to both GPUs.

            let newState = { ...gpu };
            let nextStatus: GpuState['status'] = 'idle';
            newState.currentLayerName = undefined; // Reset per step
            // Preserve temp full flag ONLY if the current step maintains it (e.g., FSDP AllGather -> Compute)
            const preserveParamsTempFull = strategy === 'fsdp' && gpu.isParamsTempFull && (details.type === 'COMPUTE' || details.type === 'UPDATE');
            newState.isParamsTempFull = preserveParamsTempFull ? true : false; // Default to false unless preserved

            // FIX: Set Activation Memory from pre-calculated profile FIRST
            newState.activationMemory = calculatedActivationMemory;

            // --- Result of PREVIOUS step --- Determine state *entering* current step
            if (prevStepDetails) {
                 // FSDP Param Gather/Discard
                 if (strategy === 'fsdp' && prevStepDetails.type === 'COMM' && prevStepDetails.operation === 'AllGather' && prevStepDetails.dataType === 'Params') newState.isParamsTempFull = true;
                 else if (strategy === 'fsdp' && prevStepDetails.type === 'MEMORY_OP' && prevStepDetails.operation === 'DiscardParams') newState.isParamsTempFull = false;
                 // DP/FSDP Gradient Communication
                 else if (strategy === 'dp' && prevStepDetails.type === 'COMM' && prevStepDetails.operation === 'AllReduce') newState.gradientMemory = MAX_GRADIENT; // DP has full grads after AllReduce
                 else if (strategy === 'fsdp' && prevStepDetails.type === 'COMM' && prevStepDetails.operation === 'ReduceScatter') newState.gradientMemory = MAX_GRADIENT / shardDenom;
                 // FIX: Apply grad clear uniformly AFTER update step for all strategies
                 if (prevStepDetails.type === 'UPDATE') {
                     newState.gradientMemory = 0;
                 }
                 // TP Specific Previous Step Logic (e.g., after activation AllReduce)
                 // Although profile handles memory size, we might note the logical state change.
                 // We don't explicitly track hidden vs sequence sharding yet, so not much changes visually here.

            } else if (clampedStep === 0) { // Handle step 0 explicitly
                 const initState = initializeGpuStates(expectedGpuCount, strategy)[gpu.id];
                 newState.paramMemory = initState.paramMemory;
                 newState.activationMemory = 0; // From profile[0]
                 newState.gradientMemory = 0;
                 newState.optStateMemory = initState.optStateMemory;
                 newState.isParamsTempFull = false;
            }

            // --- State FOR CURRENT step --- Determine effects of current step
            switch (details.type) {
                case 'INIT':
                    nextStatus = 'idle';
                    // Memory already set by initialization logic above
                    break;
                case 'COMPUTE':
                    nextStatus = 'computing';
                    newState.currentLayerName = details.layer || 'Compute';
                    if (strategy === 'tp' && details.subStep) { // TP: Add sub-step info
                        newState.currentLayerName += ` (${details.subStep})`;
                    } else if ((strategy === 'dp' || strategy === 'fsdp') && details.direction === 'forward' && newState.dataShardId) {
                        newState.currentLayerName = `B${newState.dataShardId}: Fwd-${details.layer}`;
                    } else if (details.direction === 'backward') {
                         newState.currentLayerName = `Bwd-${details.layer}`;
                         if (strategy === 'tp' && details.subStep) newState.currentLayerName += ` (${details.subStep})`;
                         // Handle gradient accumulation during backward
                         if (strategy === 'fsdp') newState.gradientMemory = MAX_GRADIENT; // FSDP: Temp full local grad before ReduceScatter
                         if (strategy === 'dp') newState.gradientMemory = MAX_GRADIENT; // DP: Accumulates full gradients locally
                         if (strategy === 'tp') newState.gradientMemory = MAX_GRADIENT / shardDenom; // TP: Accumulates sharded gradients (simplified)
                    }
                    // Preserve FSDP temp full param flag if set by previous AllGather
                    if(strategy === 'fsdp' && gpu.isParamsTempFull) newState.isParamsTempFull = true;
                    break;
                case 'GRADIENTS': // Primarily used in original DP logic
                    nextStatus = 'computing';
                    newState.currentLayerName = `Grads g_${gpu.id}`;
                    newState.gradientMemory = MAX_GRADIENT;
                    break;
                case 'COMM':
                    nextStatus = 'communicating';
                    newState.currentLayerName = `${details.operation} (${details.dataType})`;
                    // FSDP AllGather Params makes params temporarily full
                    if (strategy === 'fsdp' && details.operation === 'AllGather' && details.dataType === 'Params') newState.isParamsTempFull = true;
                    // TP communication doesn't change persistent param sharding state in this viz
                    // Handle gradient updates post-communication
                    if (details.operation === 'AllReduce' && details.dataType === 'Gradients' && strategy === 'dp') newState.gradientMemory = MAX_GRADIENT;
                    if (details.operation === 'ReduceScatter' && details.dataType === 'Gradients' && strategy === 'fsdp') newState.gradientMemory = MAX_GRADIENT / shardDenom;
                    // Activation comms are handled by the profile
                    break;
                case 'MEMORY_OP': // FSDP DiscardParams
                    nextStatus = 'idle';
                    newState.currentLayerName = 'Discard Shards';
                    newState.isParamsTempFull = false;
                    break;
                case 'UPDATE':
                    nextStatus = 'computing';
                    newState.currentLayerName = 'Optimizer';
                    // Gradient memory will be cleared *after* this step (handled in next step's logic)
                    if(strategy === 'fsdp' && gpu.isParamsTempFull) newState.isParamsTempFull = true; // Preserve flag if needed
                    break;
                case 'DONE':
                    nextStatus = 'idle';
                    newState.gradientMemory = 0; // Ensure grads are zero at the end
                    newState.isParamsTempFull = false;
                    // Activation memory handled by profile
                    break;
                default:
                    nextStatus = 'idle';
                    break;
            }
            newState.status = nextStatus;
            return newState;
        });
    });
}, [currentSimulationSteps, strategy, numGpus, totalSteps, activationMemoryProfile]); // Keep dependencies updated

  // --- Simulation Controls & Management --- UPDATED CALLBACKS
  const intervalSpeed = 1000;
  const clearSimulationInterval = useCallback(() => { if (intervalRef.current !== null) clearInterval(intervalRef.current); intervalRef.current = null; }, []);

  // Advance step function remains largely the same
  const advanceStep = useCallback((direction: 1 | -1 = 1) => {
    setCurrentStep(prev => {
        const next = prev + direction;
        const clampedNext = Math.max(0, Math.min(next, totalSteps));
        if (clampedNext !== prev) {
            if (isPlaying) {
               // If playing, let the interval handle pausing at the end
            } else {
               setIsPlaying(false); // Ensure stopped if manually stepping
               clearSimulationInterval();
            }
            updateGpuStatesForStep(clampedNext);
            return clampedNext;
        } else if (clampedNext === totalSteps && isPlaying) {
            // If we reached the end while playing, stop
            setIsPlaying(false);
            clearSimulationInterval();
        }
        return prev;
    });
  }, [totalSteps, clearSimulationInterval, updateGpuStatesForStep, isPlaying]); // Added isPlaying dependency

  const prevStep = useCallback(() => { advanceStep(-1); }, [advanceStep]);

  // Play/Pause Toggle logic (corrected)
   const handlePlayPause = useCallback(() => {
      setIsPlaying(prevIsPlaying => {
          const currentlyPlaying = !prevIsPlaying;
          if (currentlyPlaying) {
              // --- Start Playing --- START HERE
              if (currentStep >= totalSteps) {
                  return false; // Can't play if already at the end
              }
              clearSimulationInterval(); // Clear any existing interval

              // Immediately advance one step
               setCurrentStep(prev => {
                  const nextStepImmediate = Math.min(prev + 1, totalSteps);
                  updateGpuStatesForStep(nextStepImmediate);
                  // Check if we reached the end after this immediate step
                  if (nextStepImmediate < totalSteps) {
                      // Start interval ONLY if not at the end
                       intervalRef.current = window.setInterval(() => {
                           advanceStep(1);
                       }, intervalSpeed);
                  } else {
                      // Reached the end, don't start interval, ensure state is 'paused'
                      return prev; // Return current step, but isPlaying will be set to false below
                  }
                  return nextStepImmediate;
              });
               return true; // Set state to playing
          } else {
              // --- Pause Playing --- STOP HERE
              clearSimulationInterval();
              return false; // Set state to not playing
          }
      });
  }, [currentStep, totalSteps, clearSimulationInterval, advanceStep, intervalSpeed, updateGpuStatesForStep]);

  // Reset simulation - UPDATED
  const reset = useCallback(() => {
    setIsPlaying(false);
    clearSimulationInterval();
    setCurrentStep(0);
    // Initialize based on current strategy and numGpus
    setGpuStates(initializeGpuStates(numGpus, strategy));
    // Explicitly call update for step 0 to ensure correct initial state visualization
    // Need to get the freshly generated steps for the current strategy/GPU count
    let stepsForReset: StepDetail[];
    if (strategy === 'dp') stepsForReset = generateDpSteps(numGpus);
    else if (strategy === 'fsdp') stepsForReset = generateFsdpSteps(numGpus);
    else if (strategy === 'tp') stepsForReset = generateTpSteps();
    else stepsForReset = generateSingleGpuSteps();

    if (stepsForReset.length > 0) {
       const profileForReset = calculateActivationProfile(stepsForReset);
       // Manually apply state for step 0 using the reset context
       const initialGpuStatesForStep0 = initializeGpuStates(numGpus, strategy).map(gpu => ({
          ...gpu,
          activationMemory: profileForReset[0] ?? 0,
          status: 'idle',
          currentLayerName: stepsForReset[0]?.description ?? 'Init',
       }));
       setGpuStates(initialGpuStatesForStep0);
    } else {
         setGpuStates(initializeGpuStates(numGpus, strategy)); // Fallback
    }

  }, [numGpus, strategy, clearSimulationInterval]); // Removed updateGpuStatesForStep dependency as we handle step 0 manually

  // Callback to set number of GPUs - UPDATED FOR TP
   const setNumGpusCallback = useCallback((num: number) => {
        // Prevent changing GPU count if strategy dictates it
        if (strategy === 'tp' && num !== 2) {
            console.log("Cannot change GPU count for TP (fixed at 2).");
            return; // TP is fixed at 2
        }
        if (strategy === 'single' && num !== 1) {
             console.log("Cannot change GPU count for Single GPU strategy.");
            return; // Single is fixed at 1
        }
        // Ensure DP/FSDP have at least 2 GPUs
        if ((strategy === 'dp' || strategy === 'fsdp') && num < 2) {
             console.log("DP/FSDP requires at least 2 GPUs.");
             num = 2;
        }

        if (num !== numGpus) {
            setNumGpusInternal(num);
            // Reset is handled by useEffect dependency change
        }
    }, [numGpus, strategy]);

  // Callback to set strategy - UPDATED FOR TP
  const setStrategyCallback = useCallback((newStrategy: string) => {
    if (newStrategy !== strategy) {
        let nextNumGpus = numGpus; // Start with current

        if (newStrategy === 'single') {
            nextNumGpus = 1;
        } else if (newStrategy === 'tp') { // <<<--- HANDLE TP
            nextNumGpus = 2; // Force TP group size to 2
        } else if ((newStrategy === 'dp' || newStrategy === 'fsdp')) {
             // If switching to DP/FSDP, ensure at least 2 GPUs
             if (numGpus < 2) {
                  nextNumGpus = 2;
             }
        }

        // Update numGpus state *if* it needs to change
        if (nextNumGpus !== numGpus) {
            setNumGpusInternal(nextNumGpus);
        }
        // Always update the strategy state
        setStrategyInternal(newStrategy);
        // Reset will be triggered by the useEffect below due to state changes
    }
}, [numGpus, strategy]);

  // Effect to reset simulation state ONLY when strategy or numGpus changes
  const isInitialMount = useRef(true);
  useEffect(() => {
    if (isInitialMount.current) {
      isInitialMount.current = false;
      // Initial update for step 0 on mount
      updateGpuStatesForStep(0);
    } else {
      // Reset whenever strategy or numGpus changes after initial mount
      console.log(`Strategy or NumGpus changed to: ${strategy}, ${numGpus}. Resetting simulation.`);
      reset();
    }
  }, [strategy, numGpus, reset]); // Use reset callback dependency

  // Cleanup interval on unmount
  useEffect(() => {
      return () => clearSimulationInterval();
  }, [clearSimulationInterval]);

  const value = {
      currentStep,
      totalSteps,
      isPlaying,
      strategy,
      numGpus,
      gpuStates,
      stepDetails,
      play: handlePlayPause, // Use single toggle handler
      pause: handlePlayPause, // Use single toggle handler
      nextStep: () => advanceStep(1),
      prevStep,
      reset,
      setStrategy: setStrategyCallback,
      setNumGpus: setNumGpusCallback
  };

  return ( <SimulationContext.Provider value={value}>{children}</SimulationContext.Provider> );
};

// Custom hook to use the simulation context
export const useSimulation = () => {
  const context = useContext(SimulationContext);
  if (context === undefined) {
    throw new Error('useSimulation must be used within a SimulationProvider');
  }
  return context;
};
