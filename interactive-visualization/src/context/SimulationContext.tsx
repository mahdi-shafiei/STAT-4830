import React, { createContext, useState, useContext, useEffect, useRef, useCallback } from 'react';
import { generateDpSteps } from '../config/strategies/dp';
import { generateFsdpSteps } from '../config/strategies/fsdp';
import { generateSingleGpuSteps } from './singleGpuSteps';
import type { GpuState, CommOperation, CommDataType, StepDetail } from './types';

export type { GpuState, CommOperation, CommDataType, StepDetail };

const MAX_ACTIVATION = 75; const MAX_PARAM = 100; const MAX_OPTSTATE = 100; const MAX_GRADIENT = 100;
const MODEL_LAYERS = ['Embed', 'MHA', 'FFN', 'LN', 'Output'];

// Helper: Initialize GPU States - Robust version
const initializeGpuStates = (numGpus: number, strategy: string): GpuState[] => {
    // Ensure numGpus is at least 1
    const count = Math.max(1, numGpus);
    console.log(`Initializing state for ${count} GPUs, strategy: ${strategy}`);
    return Array.from({ length: count }, (_, i) => {
        let iP = 0, iO = 0, iG = 0;
        // Ensure shardDenom is at least 1 to avoid division by zero
        const shardDenom = (strategy === 'fsdp' && count > 0) ? count : 1;
        const isParallel = strategy === 'dp' || strategy === 'fsdp';

        if (strategy === 'single' || strategy === 'dp') { iP = MAX_PARAM; iO = MAX_OPTSTATE; }
        else if (strategy === 'fsdp') { iP = MAX_PARAM / shardDenom; iO = MAX_OPTSTATE / shardDenom; }

        return { id: i, paramMemory: iP, activationMemory: 0, gradientMemory: iG, optStateMemory: iO,
            status: 'idle', currentLayerName: undefined, isParamsTempFull: false,
            dataShardId: isParallel ? i + 1 : undefined, };
    });
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
  // Initialize state carefully
  const [gpuStates, setGpuStates] = useState<GpuState[]>(() => initializeGpuStates(numGpus, strategy));
  const intervalRef = useRef<number | null>(null);

  // Determine current simulation steps dynamically and safely
  const { currentSimulationSteps, totalSteps } = React.useMemo(() => {
    let steps: StepDetail[];
    try {
      if (strategy === 'dp') steps = generateDpSteps(numGpus);
      else if (strategy === 'fsdp') steps = generateFsdpSteps(numGpus);
      else steps = generateSingleGpuSteps();
    } catch (e) {
        console.error("Error generating simulation steps:", e);
        steps = [{step: 0, type: 'ERROR', description: 'Error generating steps.'}]; // Fallback
    }
    const totSteps = steps.length > 0 ? steps.length - 1 : 0;
    console.log(`Generated ${steps.length} steps for ${strategy} with ${numGpus} GPUs. Total steps index: ${totSteps}`);
    return { currentSimulationSteps: steps, totalSteps: totSteps };
  }, [strategy, numGpus]);

  // Ensure stepDetails is derived safely
  const stepDetails = (currentStep >= 0 && currentStep < currentSimulationSteps.length)
    ? currentSimulationSteps[currentStep]
    : null;

  // --- State Update Logic (with added checks) ---
  const updateGpuStatesForStep = useCallback((step: number) => {
    // Ensure steps array is valid before proceeding
    if (!currentSimulationSteps || currentSimulationSteps.length === 0) {
        console.error("Attempted to update state with invalid simulation steps.");
        return;
    }
    // Clamp step to valid range
    const clampedStep = Math.max(0, Math.min(step, totalSteps));
    const details = currentSimulationSteps[clampedStep];

    if (!details) { console.warn(`No step details found for step ${clampedStep}`); return; }

    setGpuStates(prevStates => {
        // Defensive check on prevStates length
        if (!Array.isArray(prevStates) || (prevStates.length !== numGpus && strategy !== 'single')) {
             console.warn(`GPU state array length mismatch. Re-initializing. Prev length: ${prevStates?.length}, Expected: ${numGpus}`);
             return initializeGpuStates(numGpus, strategy);
        }

        const isParallel = details.parallel === true;
        const isParamsSharded = strategy === 'fsdp'; const isGradsSharded = strategy === 'fsdp'; const isOptStatesSharded = strategy === 'fsdp';
        const shardDenom = numGpus > 0 ? numGpus : 1;

        return prevStates.map((gpu) => {
            if (!isParallel && strategy !== 'single' && gpu.id !== 0) return gpu;

            let newState = { ...gpu }; let nextStatus: GpuState['status'] = 'idle';
            newState.currentLayerName = undefined; newState.isParamsTempFull = false;

            const prevStepDetails = (clampedStep > 0 && clampedStep - 1 < currentSimulationSteps.length)
                ? currentSimulationSteps[clampedStep - 1]
                : null;

            // --- Result of PREVIOUS step ---
            if (prevStepDetails) {
                 if (strategy === 'fsdp' && prevStepDetails.type === 'COMM' && prevStepDetails.operation === 'AllGather' && prevStepDetails.dataType === 'Params') newState.isParamsTempFull = true;
                 else if (strategy === 'fsdp' && prevStepDetails.type === 'MEMORY_OP' && prevStepDetails.operation === 'DiscardParams') newState.isParamsTempFull = false;
                 else if (strategy === 'dp' && prevStepDetails.type === 'COMM' && prevStepDetails.operation === 'AllReduce') newState.gradientMemory = MAX_GRADIENT * 0.6;
                 else if (strategy === 'fsdp' && prevStepDetails.type === 'COMM' && prevStepDetails.operation === 'ReduceScatter') newState.gradientMemory = MAX_GRADIENT / shardDenom;
                 else if (prevStepDetails.type === 'UPDATE' && (strategy === 'dp' || strategy === 'fsdp')) newState.gradientMemory = 0; // Uniformly clear grads
            } else if (clampedStep === 0) { // Ensure reset on step 0
                 Object.assign(newState, initializeGpuStates(numGpus, strategy)[gpu.id]);
                 return newState;
            }

            // --- State FOR CURRENT step ---
            switch (details.type) {
                case 'INIT': nextStatus = 'idle'; newState.activationMemory = 0; newState.gradientMemory = 0; newState.isParamsTempFull = false; break;
                case 'COMPUTE': /* ... unchanged compute logic ... */
                    nextStatus = 'computing'; newState.currentLayerName = details.layer || 'Compute';
                    if ((strategy === 'dp' || strategy === 'fsdp') && details.direction === 'forward' && newState.dataShardId) newState.currentLayerName = `B${newState.dataShardId}: Fwd-${details.layer}`;
                    else if (details.direction === 'backward') newState.currentLayerName = `Bwd-${details.layer}`;
                    if (details.direction === 'forward') { const fSteps = currentSimulationSteps.filter(s => s.type === 'COMPUTE' && s.direction === 'forward').length; const growth = fSteps > 0 ? MAX_ACTIVATION / fSteps : MAX_ACTIVATION; newState.activationMemory = Math.min(gpu.activationMemory + growth, MAX_ACTIVATION); }
                    else if (details.direction === 'backward') { if (strategy === 'fsdp') newState.gradientMemory = MAX_GRADIENT; newState.activationMemory = Math.max(0, gpu.activationMemory - MAX_ACTIVATION / MODEL_LAYERS.length); }
                    if(strategy === 'fsdp' && gpu.isParamsTempFull) newState.isParamsTempFull = true; // Preserve flag
                    break;
                case 'GRADIENTS': nextStatus = 'computing'; newState.currentLayerName = `Grads g${gpu.id}`; newState.gradientMemory = MAX_GRADIENT; newState.activationMemory = 0; break;
                case 'COMM': /* ... unchanged COMM logic ... */
                    nextStatus = 'communicating'; newState.currentLayerName = `${details.operation} (${details.dataType})`;
                    if (strategy === 'fsdp' && details.operation === 'AllGather' && details.dataType === 'Params') newState.isParamsTempFull = true; // Set during COMM
                    break;
                 case 'MEMORY_OP': /* ... unchanged MEMORY_OP logic ... */
                     nextStatus = 'idle'; newState.currentLayerName = 'Discard Shards'; newState.isParamsTempFull = false; // Explicitly unset
                     break;
                case 'UPDATE': nextStatus = 'computing'; newState.currentLayerName = 'Optimizer'; break;
                case 'DONE': nextStatus = 'idle'; newState.gradientMemory = 0; newState.activationMemory = 0; newState.isParamsTempFull = false; break;
                default: nextStatus = 'idle'; break;
            }
            newState.status = nextStatus;
            return newState;
        });
    });
  }, [currentSimulationSteps, strategy, numGpus, totalSteps]); // Added totalSteps dependency

  // --- Simulation Controls & Management ---
  const intervalSpeed = 1000;
  const clearSimulationInterval = useCallback(() => { if (intervalRef.current !== null) clearInterval(intervalRef.current); intervalRef.current = null; }, []);
  const advanceStep = useCallback((direction: 1 | -1 = 1) => { setCurrentStep(prev => { const next = prev + direction; const clampedNext = Math.max(0, Math.min(next, totalSteps)); if (clampedNext !== prev) { setIsPlaying(false); clearSimulationInterval(); updateGpuStatesForStep(clampedNext); return clampedNext; } return prev; }); }, [totalSteps, clearSimulationInterval, updateGpuStatesForStep]);
  const prevStep = useCallback(() => { advanceStep(-1); }, [advanceStep]);
  const play = useCallback(() => { if (isPlaying || currentStep >= totalSteps) return; setIsPlaying(true); clearSimulationInterval(); if (currentStep < totalSteps) { advanceStep(1); setCurrentStep(prevStep => { if (prevStep < totalSteps) { intervalRef.current = window.setInterval(() => advanceStep(1), intervalSpeed); } else { setIsPlaying(false); } return prevStep; }); } else { setIsPlaying(false); } }, [isPlaying, currentStep, totalSteps, clearSimulationInterval, advanceStep, intervalSpeed]);
  const pause = useCallback(() => { if (!isPlaying) return; setIsPlaying(false); clearSimulationInterval(); }, [isPlaying, clearSimulationInterval]);
  // Reset now correctly uses state setters to trigger useEffect
  const reset = useCallback(() => { setIsPlaying(false); clearSimulationInterval(); setCurrentStep(0); setGpuStates(initializeGpuStates(numGpus, strategy)); /* State update triggered by useEffect */ }, [numGpus, strategy, clearSimulationInterval]);
  // Setters now just update state, useEffect handles reset
  const setNumGpusCallback = useCallback((num: number) => { if (num !== numGpus && strategy !== 'single') { setNumGpusInternal(num); } }, [numGpus, strategy]);
  const setStrategyCallback = useCallback((newStrategy: string) => { if (newStrategy !== strategy) { const nextNumGpus = newStrategy === 'single' ? 1 : (numGpus >= 2 ? numGpus : 2); setNumGpusInternal(nextNumGpus); setStrategyInternal(newStrategy); } }, [numGpus, strategy]);

  // Effect to reset simulation state when strategy or numGpus changes
  useEffect(() => { reset(); }, [strategy, numGpus, reset]); // Use reset in dependency

  useEffect(() => clearSimulationInterval, [clearSimulationInterval]); // Cleanup

  const value = { currentStep, totalSteps, isPlaying, strategy, numGpus, gpuStates, stepDetails, play, pause, nextStep: () => advanceStep(1), prevStep, reset, setStrategy: setStrategyCallback, setNumGpus: setNumGpusCallback };
  return ( <SimulationContext.Provider value={value}>{children}</SimulationContext.Provider> );
};
export const useSimulation = () => { const context = useContext(SimulationContext); if (context === undefined) throw new Error('useSimulation must be used within a SimulationProvider'); return context; };
