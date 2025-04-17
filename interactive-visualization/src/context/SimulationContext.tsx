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

// --- Helper: Pre-calculate Activation Memory Profile (REFINED for TP) ---
const calculateActivationProfile = (steps: StepDetail[]): number[] => {
    const profile: number[] = Array(steps.length).fill(0);
    if (!steps || steps.length === 0) return profile;

    const activationMemoryCosts: Record<string, number> = {}; // Cost to store the *full* activation tensor
    const activationReleaseStep: Record<string, number> = {};
    const costPerLayer = MODEL_LAYERS.length > 0 ? MAX_ACTIVATION / MODEL_LAYERS.length : 0;
    MODEL_LAYERS.forEach(layer => activationMemoryCosts[layer] = costPerLayer);
    // Input cost is 0
    activationMemoryCosts['Input'] = 0;
    const Ntp = 2; // Fixed TP size for calculation consistency
    const currentStrategy = steps[0]?.strategy || 'unknown';

    // Find release step (step *after* activation is consumed)
    steps.forEach((step, index) => { // Added index for clarity
        if (step.type === 'COMPUTE' && step.direction === 'backward' && step.activationConsumedLayer !== null) { // Input is null
            const layerToRelease = step.activationConsumedLayer as string; // Type assertion
            const releaseAfterStepIndex = index; // Activation is needed *during* this step
            // Release happens *after* this step completes. Update only if this is an earlier release point.
            if (!activationReleaseStep[layerToRelease] || activationReleaseStep[layerToRelease] > releaseAfterStepIndex + 1) {
                activationReleaseStep[layerToRelease] = releaseAfterStepIndex + 1;
            }
        }
        // Handle DP/Other strategies where GRADIENTS or DONE implies release
        if ((step.type === 'GRADIENTS' && step.strategy === 'dp') || step.type === 'DONE') {
             MODEL_LAYERS.forEach(layer => {
                // Release after step, unless it's DONE (release during)
                 const releasePoint = index + (step.type === 'DONE' ? 0 : 1);
                 if (!activationReleaseStep[layer] || activationReleaseStep[layer] > releasePoint ) {
                     activationReleaseStep[layer] = releasePoint;
                 }
             });
             // Also release Input if held
              const releasePoint = index + (step.type === 'DONE' ? 0 : 1);
              if (!activationReleaseStep['Input'] || activationReleaseStep['Input'] > releasePoint ) {
                     activationReleaseStep['Input'] = releasePoint;
              }
        }
    });
    // Ensure last step always releases everything not already scheduled
    const lastStepIndex = steps.length > 0 ? steps.length - 1 : 0;
     MODEL_LAYERS.forEach(layer => {
         if (!activationReleaseStep[layer]) activationReleaseStep[layer] = lastStepIndex; // Release at end
     });
     if (!activationReleaseStep['Input']) activationReleaseStep['Input'] = lastStepIndex;


    // Calculate profile step-by-step
    let memoryHeld = 0;
    const activeActivations = new Map<string, number>(); // Store activation name and its *current* cost (full or sharded)

    for (let i = 0; i < steps.length; i++) {
        const currentStepDetails = steps[i];

        // --- Release Memory FIRST ---
        // Check activations against their release step for *this* step i
        const releasedLayers: string[] = [];
        activeActivations.forEach((cost, layerName) => {
            // Release if the release step is *this* step or earlier (robustness)
            if (activationReleaseStep[layerName] !== undefined && activationReleaseStep[layerName] <= i) {
                // console.log(`Step ${i} (${currentStrategy}): Releasing ${layerName} (cost ${cost.toFixed(1)}), release scheduled at ${activationReleaseStep[layerName]}`);
                memoryHeld -= cost;
                releasedLayers.push(layerName);
            }
        });
        releasedLayers.forEach(layerName => activeActivations.delete(layerName));


        // --- Add Memory produced by PREVIOUS step's output ---
        // We look at the *previous* step (i-1) to see what activation it produced, which is now stored.
        const prevStepDetails = i > 0 ? steps[i - 1] : null;
        if (prevStepDetails?.activationProduced) {
            const layerName = prevStepDetails.activationProduced;

            if (!activeActivations.has(layerName)) { // Only add if not already active
                 let costToAdd = activationMemoryCosts[layerName] ?? 0;

                // --- TP Sharding Logic ---
                // If the *producing* step (prevStepDetails) was a TP compute step that outputs a hidden-sharded activation
                // (e.g., ColParallel, LocalAttention), store sharded cost.
                // If it was a TP COMM step (AllReduce), it produces a full replicated activation.
                 const producingStrategy = prevStepDetails.strategy;
                 const producingType = prevStepDetails.type;
                 const producingTpExecution = prevStepDetails.tpExecutionType as string | undefined; // Get TP execution type

                 if (producingStrategy === 'tp' && producingType === 'COMPUTE' &&
                    (producingTpExecution === 'ColumnParallel' ||
                     producingTpExecution === 'LocalAttention' ||
                     (producingTpExecution === 'RowParallel' && !prevStepDetails.subStep?.includes('Reduce')) // RowParallel *compute* produces sharded Z_k
                    )) {
                     costToAdd /= Ntp; // Store only the sharded activation cost
                     // console.log(`Step ${i} (${currentStrategy}): Adding SHARDED activation for ${layerName} (cost ${costToAdd.toFixed(1)}) produced by prev step ${i-1} (${producingTpExecution})`);
                 } else {
                     // console.log(`Step ${i} (${currentStrategy}): Adding FULL activation for ${layerName} (cost ${costToAdd.toFixed(1)}) produced by prev step ${i-1} (${producingStrategy}, ${producingType}, ${producingTpExecution})`);
                 }
                 // Else (TP AllReduce comm, non-TP steps) store full cost

                activeActivations.set(layerName, costToAdd);
                memoryHeld += costToAdd;

            } else {
                // Activation already present? This shouldn't happen if logic is correct.
                // If it does, maybe update cost if sharding changes? For now, assume it doesn't overwrite.
                 console.warn(`Step ${i}: Activation ${layerName} produced by step ${i-1} was already active?`);
            }
        }

        // Assign memory for the *current* step i
        profile[i] = Math.max(0, Math.min(memoryHeld, MAX_ACTIVATION));

        // Explicitly ensure step 0 and DONE are 0 if calculation leads elsewhere
        if (currentStepDetails?.type === 'INIT') profile[i] = 0;
        // Ensure DONE step is 0 (release logic should handle this, but enforce)
        if (currentStepDetails?.type === 'DONE') {
             profile[i] = 0;
             activeActivations.clear(); // Clear all remaining activations at DONE
             memoryHeld = 0;
        }
    }
    console.log(`Activation Profile (${currentStrategy}):`, profile.map(p => p.toFixed(1)));
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
