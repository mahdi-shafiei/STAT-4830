import React, { createContext, useState, useContext, useEffect, useRef, useCallback } from 'react';
import { generateDpSteps } from '../config/strategies/dp';
import { generateFsdpSteps } from '../config/strategies/fsdp';
import { generateSingleGpuSteps } from './singleGpuSteps';
import type { GpuState, CommOperation, CommDataType, StepDetail } from './types';

export type { GpuState, CommOperation, CommDataType, StepDetail };

const MAX_ACTIVATION = 100; const MAX_PARAM = 100; const MAX_OPTSTATE = 100; const MAX_GRADIENT = 100;
const MODEL_LAYERS = ['Embed', 'MHA', 'FFN', 'LN', 'Output']; // Must match layers used in step generators

// --- Helper: Initialize GPU States ---
const initializeGpuStates = (numGpus: number, strategy: string): GpuState[] => { /* ... unchanged ... */
    const count = Math.max(1, numGpus); console.log(`Initializing state for ${count} GPUs, strategy: ${strategy}`);
    return Array.from({ length: count }, (_, i) => {
        let iP = 0, iO = 0, iG = 0; const shardDenom = (strategy === 'fsdp' && count > 0) ? count : 1; const isParallel = strategy === 'dp' || strategy === 'fsdp';
        if (strategy === 'single' || strategy === 'dp') { iP = MAX_PARAM; iO = MAX_OPTSTATE; } else if (strategy === 'fsdp') { iP = MAX_PARAM / shardDenom; iO = MAX_OPTSTATE / shardDenom; }
        return { id: i, paramMemory: iP, activationMemory: 0, gradientMemory: iG, optStateMemory: iO, status: 'idle', currentLayerName: undefined, isParamsTempFull: false, dataShardId: isParallel ? i + 1 : undefined, };
    });
};

// --- Helper: Pre-calculate Activation Memory Profile ---
const calculateActivationProfile = (steps: StepDetail[]): number[] => {
    const profile: number[] = Array(steps.length).fill(0);
    const activationMemoryCosts: Record<string, number> = {}; // Store memory cost per activation layer name
    const activationReleaseStep: Record<string, number> = {}; // Store step when activation is released

    // Assign memory cost (simplified: proportional to total)
    const costPerLayer = MAX_ACTIVATION / MODEL_LAYERS.length;
    MODEL_LAYERS.forEach(layer => activationMemoryCosts[layer] = costPerLayer);
    activationMemoryCosts['Input'] = 0; // Input doesn't add cost

    // Find when each activation is released (consumed by the *next* layer's backward step)
    steps.forEach(step => {
        if (step.type === 'COMPUTE' && step.direction === 'backward' && step.activationConsumedLayer) {
             // Mark the step number where this activation is no longer needed
             activationReleaseStep[step.activationConsumedLayer] = step.step;
        }
        // Handle DP gradient step releasing all activations conceptually
        if (step.type === 'GRADIENTS' && step.strategy === 'dp') {
            MODEL_LAYERS.forEach(layer => {
                if (!activationReleaseStep[layer] || activationReleaseStep[layer] > step.step) {
                    activationReleaseStep[layer] = step.step;
                }
            });
        }
        // Final DONE step releases anything remaining
        if (step.type === 'DONE') {
             MODEL_LAYERS.forEach(layer => {
                if (!activationReleaseStep[layer]) {
                    activationReleaseStep[layer] = step.step;
                }
             });
        }
    });
    console.log("Activation Release Steps:", activationReleaseStep);


    // Calculate cumulative memory profile
    let currentActivations: Record<string, boolean> = {}; // Track active activations
    steps.forEach((step, i) => {
        // Add memory for activations produced by this step
        if (step.type === 'COMPUTE' && step.direction === 'forward' && step.activationProduced) {
            currentActivations[step.activationProduced] = true;
        }

        // Calculate total memory held at the END of this step
        let memoryAtStepEnd = 0;
        for (const layerName in currentActivations) {
            // Include memory if it hasn't been released yet (release happens *after* the consuming step)
            if (!activationReleaseStep[layerName] || activationReleaseStep[layerName] >= step.step) {
                 memoryAtStepEnd += activationMemoryCosts[layerName] ?? 0;
            } else {
                // Activation was released at or before this step, remove it for next step calc
                 delete currentActivations[layerName];
                 console.log(`Step ${step.step}: Releasing activation for ${layerName}`);
            }
        }
         profile[i] = Math.min(memoryAtStepEnd, MAX_ACTIVATION); // Cap at max

        // Reset memory at the very start or very end
        if (step.type === 'INIT') profile[i] = 0;
        if (step.type === 'DONE') profile[i] = 0; // Ensure cleared at the end
    });
    console.log("Activation Memory Profile:", profile);
    return profile;
};


// --- Context Definition ---
interface SimulationState { currentStep: number; totalSteps: number; isPlaying: boolean; strategy: string; numGpus: number; gpuStates: GpuState[]; stepDetails: StepDetail | null; }
interface SimulationContextProps extends SimulationState { play: () => void; pause: () => void; nextStep: () => void; prevStep: () => void; reset: () => void; setStrategy: (strategy: string) => void; setNumGpus: (num: number) => void; }
const SimulationContext = createContext<SimulationContextProps | undefined>(undefined);

// --- Context Provider ---
export const SimulationProvider: React.FC<{ children: React.ReactNode; initialNumGpus?: number }> = ({ children, initialNumGpus = 1 }) => {
  const [strategy, setStrategyInternal] = useState(() => initialNumGpus > 1 ? 'fsdp' : 'single');
  const [numGpus, setNumGpusInternal] = useState(() => strategy === 'single' ? 1 : Math.max(2, initialNumGpus));
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [gpuStates, setGpuStates] = useState<GpuState[]>(() => initializeGpuStates(numGpus, strategy));
  const intervalRef = useRef<number | null>(null);

  // Generate steps AND activation profile
  const { currentSimulationSteps, totalSteps, activationMemoryProfile } = React.useMemo(() => {
    let steps: StepDetail[];
    if (strategy === 'dp') steps = generateDpSteps(numGpus);
    else if (strategy === 'fsdp') steps = generateFsdpSteps(numGpus);
    else steps = generateSingleGpuSteps();
    const profile = calculateActivationProfile(steps); // Calculate profile
    const totSteps = steps.length > 0 ? steps.length - 1 : 0;
    return { currentSimulationSteps: steps, totalSteps: totSteps, activationMemoryProfile: profile };
  }, [strategy, numGpus]); // Regenerate if strategy or GPU count changes

  const stepDetails = currentSimulationSteps[currentStep] || null;

  // --- State Update Logic ---
  const updateGpuStatesForStep = useCallback((step: number) => {
    if (!currentSimulationSteps || currentSimulationSteps.length === 0) return;
    const clampedStep = Math.max(0, Math.min(step, totalSteps)); // Use totalSteps from useMemo
    const details = currentSimulationSteps[clampedStep];
    if (!details) return;

    // Get pre-calculated activation memory for this step
    const calculatedActivationMemory = activationMemoryProfile[clampedStep] ?? 0;

    setGpuStates(prevStates => {
        if (!Array.isArray(prevStates) || (prevStates.length !== numGpus && strategy !== 'single')) return initializeGpuStates(numGpus, strategy);
        const isParallel = details.parallel === true; const isParamsSharded = strategy === 'fsdp'; const isGradsSharded = strategy === 'fsdp'; const isOptStatesSharded = strategy === 'fsdp'; const shardDenom = numGpus > 0 ? numGpus : 1;
        const prevStepDetails = (clampedStep > 0) ? currentSimulationSteps[clampedStep - 1] : null;

        return prevStates.map((gpu) => {
            if (!isParallel && strategy !== 'single' && gpu.id !== 0) return gpu;
            let newState = { ...gpu }; let nextStatus: GpuState['status'] = 'idle'; newState.currentLayerName = undefined; newState.isParamsTempFull = false;

            // Set Activation Memory based on pre-calculated profile for the *current* step
            newState.activationMemory = calculatedActivationMemory;

            // Result of PREVIOUS step completion
            if (prevStepDetails) {
                 if (strategy === 'fsdp' && prevStepDetails.type === 'COMM' && prevStepDetails.operation === 'AllGather' && prevStepDetails.dataType === 'Params') newState.isParamsTempFull = true;
                 else if (strategy === 'fsdp' && prevStepDetails.type === 'MEMORY_OP' && prevStepDetails.operation === 'DiscardParams') newState.isParamsTempFull = false;
                 else if (strategy === 'dp' && prevStepDetails.type === 'COMM' && prevStepDetails.operation === 'AllReduce') newState.gradientMemory = MAX_GRADIENT * 0.6;
                 else if (strategy === 'fsdp' && prevStepDetails.type === 'COMM' && prevStepDetails.operation === 'ReduceScatter') newState.gradientMemory = MAX_GRADIENT / shardDenom;
                 else if (prevStepDetails.type === 'UPDATE' && (isParallel || strategy === 'single')) newState.gradientMemory = 0;
            } else if (clampedStep === 0) { return initializeGpuStates(numGpus, strategy)[gpu.id]; } // Reset for step 0

            // State FOR CURRENT step
            switch (details.type) {
                case 'INIT': nextStatus = 'idle'; newState.gradientMemory = 0; newState.isParamsTempFull = false; break; // Activation already set by profile
                case 'COMPUTE':
                    nextStatus = 'computing'; newState.currentLayerName = details.layer || 'Compute';
                    if ((strategy === 'dp' || strategy === 'fsdp') && details.direction === 'forward' && newState.dataShardId) newState.currentLayerName = `B${newState.dataShardId}: Fwd-${details.layer}`;
                    else if (details.direction === 'backward') newState.currentLayerName = `Bwd-${details.layer}`;
                    if (details.direction === 'backward' && strategy === 'fsdp') newState.gradientMemory = MAX_GRADIENT; // Temp full local gradient
                    if(strategy === 'fsdp' && gpu.isParamsTempFull) newState.isParamsTempFull = true; // Preserve flag
                    break;
                case 'GRADIENTS': nextStatus = 'computing'; newState.currentLayerName = `Grads g_${gpu.id}`; newState.gradientMemory = MAX_GRADIENT; break; // Activation set by profile
                case 'COMM': nextStatus = 'communicating'; newState.currentLayerName = `${details.operation} (${details.dataType})`; if (strategy === 'fsdp' && details.operation === 'AllGather' && details.dataType === 'Params') newState.isParamsTempFull = true; break;
                case 'MEMORY_OP': nextStatus = 'idle'; newState.currentLayerName = 'Discard Shards'; newState.isParamsTempFull = false; break;
                case 'UPDATE': nextStatus = 'computing'; newState.currentLayerName = 'Optimizer'; break;
                case 'DONE': nextStatus = 'idle'; newState.gradientMemory = 0; newState.isParamsTempFull = false; break; // Activation set by profile (should be 0)
                default: nextStatus = 'idle'; break;
            }
            newState.status = nextStatus;
            // Ensure memory doesn't visually dip below zero if calculation is slightly off
            newState.activationMemory = Math.max(0, newState.activationMemory);
            return newState;
        });
    });
  }, [currentSimulationSteps, strategy, numGpus, totalSteps, activationMemoryProfile]); // Added profile dependency

  // --- Simulation Controls & Management ---
  const intervalSpeed = 1000;
  const clearSimulationInterval = useCallback(() => { if (intervalRef.current !== null) clearInterval(intervalRef.current); intervalRef.current = null; }, []);
  const advanceStep = useCallback((direction: 1 | -1 = 1) => { setCurrentStep(prev => { const next = prev + direction; const clampedNext = Math.max(0, Math.min(next, totalSteps)); if (clampedNext !== prev) { setIsPlaying(false); clearSimulationInterval(); updateGpuStatesForStep(clampedNext); return clampedNext; } return prev; }); }, [totalSteps, clearSimulationInterval, updateGpuStatesForStep]);
  const prevStep = useCallback(() => { advanceStep(-1); }, [advanceStep]);
  const play = useCallback(() => { if (isPlaying || currentStep >= totalSteps) return; setIsPlaying(true); clearSimulationInterval(); if (currentStep < totalSteps) { advanceStep(1); setCurrentStep(prevStep => { if (prevStep < totalSteps) { intervalRef.current = window.setInterval(() => advanceStep(1), intervalSpeed); } else { setIsPlaying(false); } return prevStep; }); } else { setIsPlaying(false); } }, [isPlaying, currentStep, totalSteps, clearSimulationInterval, advanceStep, intervalSpeed]);
  const pause = useCallback(() => { if (!isPlaying) return; setIsPlaying(false); clearSimulationInterval(); }, [isPlaying, clearSimulationInterval]);
  const reset = useCallback(() => { setIsPlaying(false); clearSimulationInterval(); setCurrentStep(0); setGpuStates(initializeGpuStates(numGpus, strategy)); }, [numGpus, strategy, clearSimulationInterval]); // Removed state setters, useEffect handles this
  // FIX: Correct logic for setting numGpus for single strategy
  const setNumGpusCallback = useCallback((num: number) => { if (num !== numGpus && strategy !== 'single') { setNumGpusInternal(num); /* useEffect handles reset */ } }, [numGpus, strategy]);
  const setStrategyCallback = useCallback((newStrategy: string) => { if (newStrategy !== strategy) { const nextNumGpus = newStrategy === 'single' ? 1 : (numGpus >= 2 ? numGpus : 2); setNumGpusInternal(nextNumGpus); setStrategyInternal(newStrategy); } }, [numGpus, strategy]);

  // Effect to reset simulation state ONLY when strategy or numGpus changes
  const isInitialMount = useRef(true);
  useEffect(() => { if (isInitialMount.current) { isInitialMount.current = false; updateGpuStatesForStep(0); } else { reset(); } }, [strategy, numGpus, reset]); // Use reset callback

  useEffect(() => clearSimulationInterval, [clearSimulationInterval]); // Cleanup

  const value = { currentStep, totalSteps, isPlaying, strategy, numGpus, gpuStates, stepDetails, play, pause, nextStep: () => advanceStep(1), prevStep, reset, setStrategy: setStrategyCallback, setNumGpus: setNumGpusCallback };
  return ( <SimulationContext.Provider value={value}>{children}</SimulationContext.Provider> );
};
export const useSimulation = () => { const context = useContext(SimulationContext); if (context === undefined) throw new Error('useSimulation must be used within a SimulationProvider'); return context; };
