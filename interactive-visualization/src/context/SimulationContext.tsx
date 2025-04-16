import React, { createContext, useState, useContext, useEffect, useRef, useCallback } from 'react';
import { generateDpSteps } from '../config/strategies/dp';
import { generateFsdpSteps } from '../config/strategies/fsdp';
import { generateSingleGpuSteps } from './singleGpuSteps';
import type { GpuState, CommOperation, CommDataType, StepDetail } from './types';

export type { GpuState, CommOperation, CommDataType, StepDetail };

const MAX_ACTIVATION = 75; const MAX_PARAM = 100; const MAX_OPTSTATE = 100; const MAX_GRADIENT = 100;
const MODEL_LAYERS = ['Embed', 'MHA', 'FFN', 'LN', 'Output']; // Use this consistently

const initializeGpuStates = (numGpus: number, strategy: string): GpuState[] => { /* ... unchanged ... */
    const count = Math.max(1, numGpus); console.log(`Initializing state for ${count} GPUs, strategy: ${strategy}`);
    return Array.from({ length: count }, (_, i) => {
        let iP = 0, iO = 0, iG = 0; const shardDenom = (strategy === 'fsdp' && count > 0) ? count : 1; const isParallel = strategy === 'dp' || strategy === 'fsdp';
        if (strategy === 'single' || strategy === 'dp') { iP = MAX_PARAM; iO = MAX_OPTSTATE; } else if (strategy === 'fsdp') { iP = MAX_PARAM / shardDenom; iO = MAX_OPTSTATE / shardDenom; }
        return { id: i, paramMemory: iP, activationMemory: 0, gradientMemory: iG, optStateMemory: iO, status: 'idle', currentLayerName: undefined, isParamsTempFull: false, dataShardId: isParallel ? i + 1 : undefined, };
    });
};

interface SimulationState { /* ... same ... */
  currentStep: number; totalSteps: number; isPlaying: boolean; strategy: string; numGpus: number; gpuStates: GpuState[]; stepDetails: StepDetail | null;
}
interface SimulationContextProps extends SimulationState { /* ... same ... */
  play: () => void; pause: () => void; nextStep: () => void; prevStep: () => void; reset: () => void; setStrategy: (strategy: string) => void; setNumGpus: (num: number) => void;
}
const SimulationContext = createContext<SimulationContextProps | undefined>(undefined);

export const SimulationProvider: React.FC<{ children: React.ReactNode; initialNumGpus?: number }> = ({ children, initialNumGpus = 1 }) => {
  // Determine initial strategy based on initialNumGpus
  const getInitialStrategy = (gpus: number) => gpus > 1 ? 'fsdp' : 'single';
  const initialStrategy = getInitialStrategy(initialNumGpus);
  const validatedInitialNumGpus = initialStrategy === 'single' ? 1 : Math.max(2, initialNumGpus);

  const [strategy, setStrategyInternal] = useState(initialStrategy);
  const [numGpus, setNumGpusInternal] = useState(validatedInitialNumGpus);
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [gpuStates, setGpuStates] = useState<GpuState[]>(() => initializeGpuStates(numGpus, strategy));
  const intervalRef = useRef<number | null>(null);

  // Generate steps based on current state
  const { currentSimulationSteps, totalSteps } = React.useMemo(() => {
    let steps: StepDetail[]; let totSteps = 0;
    if (strategy === 'dp') steps = generateDpSteps(numGpus);
    else if (strategy === 'fsdp') steps = generateFsdpSteps(numGpus);
    else steps = generateSingleGpuSteps();
    totSteps = steps.length > 0 ? steps.length - 1 : 0;
    return { currentSimulationSteps: steps, totalSteps: totSteps };
  }, [strategy, numGpus]);

  const stepDetails = currentSimulationSteps[currentStep] || null;

  // State Update Logic (with fix for DP grad clear)
  const updateGpuStatesForStep = useCallback((step: number) => {
    if (!currentSimulationSteps || currentSimulationSteps.length === 0) return;
    const clampedStep = Math.max(0, Math.min(step, totalSteps));
    const details = currentSimulationSteps[clampedStep];
    if (!details) { console.warn(`No step details for step ${clampedStep}`); return; }

    setGpuStates(prevStates => {
        if (!Array.isArray(prevStates) || (prevStates.length !== numGpus && strategy !== 'single')) return initializeGpuStates(numGpus, strategy);
        const isParallel = details.parallel === true;
        const isParamsSharded = strategy === 'fsdp'; const isGradsSharded = strategy === 'fsdp'; const isOptStatesSharded = strategy === 'fsdp';
        const shardDenom = numGpus > 0 ? numGpus : 1;
        const prevStepDetails = (clampedStep > 0) ? currentSimulationSteps[clampedStep - 1] : null;

        return prevStates.map((gpu) => {
            if (!isParallel && strategy !== 'single' && gpu.id !== 0) return gpu;
            let newState = { ...gpu }; let nextStatus: GpuState['status'] = 'idle';
            newState.currentLayerName = undefined; newState.isParamsTempFull = false;

            // Result of PREVIOUS step
            if (prevStepDetails) {
                 if (strategy === 'fsdp' && prevStepDetails.type === 'COMM' && prevStepDetails.operation === 'AllGather' && prevStepDetails.dataType === 'Params') newState.isParamsTempFull = true;
                 else if (strategy === 'fsdp' && prevStepDetails.type === 'MEMORY_OP' && prevStepDetails.operation === 'DiscardParams') newState.isParamsTempFull = false;
                 else if (strategy === 'dp' && prevStepDetails.type === 'COMM' && prevStepDetails.operation === 'AllReduce') newState.gradientMemory = MAX_GRADIENT * 0.6;
                 else if (strategy === 'fsdp' && prevStepDetails.type === 'COMM' && prevStepDetails.operation === 'ReduceScatter') newState.gradientMemory = MAX_GRADIENT / shardDenom;
                 // FIX: Clear grads AFTER update for involved GPUs (all in parallel case)
                 else if (prevStepDetails.type === 'UPDATE' && (isParallel || strategy === 'single')) {
                     newState.gradientMemory = 0;
                 }
            } else if (clampedStep === 0) { return initializeGpuStates(numGpus, strategy)[gpu.id]; }

            // State FOR CURRENT step
            switch (details.type) {
                case 'INIT': nextStatus = 'idle'; newState.activationMemory = 0; newState.gradientMemory = 0; newState.isParamsTempFull = false; break;
                case 'COMPUTE':
                    nextStatus = 'computing'; newState.currentLayerName = details.layer || 'Compute';
                    if ((strategy === 'dp' || strategy === 'fsdp') && details.direction === 'forward' && newState.dataShardId) newState.currentLayerName = `B${newState.dataShardId}: Fwd-${details.layer}`;
                    else if (details.direction === 'backward') newState.currentLayerName = `Bwd-${details.layer}`;
                    if (details.direction === 'forward') { const fSteps = currentSimulationSteps.filter(s => s.type === 'COMPUTE' && s.direction === 'forward').length; const growth = fSteps > 0 ? MAX_ACTIVATION / fSteps : MAX_ACTIVATION; newState.activationMemory = Math.min(gpu.activationMemory + growth, MAX_ACTIVATION); }
                    else if (details.direction === 'backward') { if (strategy === 'fsdp') newState.gradientMemory = MAX_GRADIENT; newState.activationMemory = Math.max(0, gpu.activationMemory - MAX_ACTIVATION / MODEL_LAYERS.length); }
                    if(strategy === 'fsdp' && gpu.isParamsTempFull) newState.isParamsTempFull = true;
                    break;
                case 'GRADIENTS': nextStatus = 'computing'; newState.currentLayerName = `Grads g_${gpu.id}`; newState.gradientMemory = MAX_GRADIENT; newState.activationMemory = 0; break; // Use g_k
                case 'COMM': nextStatus = 'communicating'; newState.currentLayerName = `${details.operation} (${details.dataType})`; if (strategy === 'fsdp' && details.operation === 'AllGather' && details.dataType === 'Params') newState.isParamsTempFull = true; break;
                case 'MEMORY_OP': nextStatus = 'idle'; newState.currentLayerName = 'Discard Shards'; newState.isParamsTempFull = false; break;
                case 'UPDATE': nextStatus = 'computing'; newState.currentLayerName = 'Optimizer'; break;
                case 'DONE': nextStatus = 'idle'; newState.gradientMemory = 0; newState.activationMemory = 0; newState.isParamsTempFull = false; break;
                default: nextStatus = 'idle'; break;
            }
            newState.status = nextStatus;
            return newState;
        });
    });
  }, [currentSimulationSteps, strategy, numGpus, totalSteps]); // Added totalSteps

  // --- Simulation Controls & Management ---
  const intervalSpeed = 1000;
  const clearSimulationInterval = useCallback(() => { if (intervalRef.current !== null) clearInterval(intervalRef.current); intervalRef.current = null; }, []);
  const advanceStep = useCallback((direction: 1 | -1 = 1) => { setCurrentStep(prev => { const next = prev + direction; const clampedNext = Math.max(0, Math.min(next, totalSteps)); if (clampedNext !== prev) { setIsPlaying(false); clearSimulationInterval(); updateGpuStatesForStep(clampedNext); return clampedNext; } return prev; }); }, [totalSteps, clearSimulationInterval, updateGpuStatesForStep]);
  const prevStep = useCallback(() => { advanceStep(-1); }, [advanceStep]);
  const play = useCallback(() => { if (isPlaying || currentStep >= totalSteps) return; setIsPlaying(true); clearSimulationInterval(); if (currentStep < totalSteps) { advanceStep(1); setCurrentStep(prevStep => { if (prevStep < totalSteps) { intervalRef.current = window.setInterval(() => advanceStep(1), intervalSpeed); } else { setIsPlaying(false); } return prevStep; }); } else { setIsPlaying(false); } }, [isPlaying, currentStep, totalSteps, clearSimulationInterval, advanceStep, intervalSpeed]);
  const pause = useCallback(() => { if (!isPlaying) return; setIsPlaying(false); clearSimulationInterval(); }, [isPlaying, clearSimulationInterval]);
  const reset = useCallback((currentNumGpus = numGpus, currentStrategy = strategy) => { setIsPlaying(false); clearSimulationInterval(); setCurrentStep(0); setGpuStates(initializeGpuStates(currentNumGpus, currentStrategy)); }, [numGpus, strategy, clearSimulationInterval]);
  // FIX: Ensure setNumGpusInternal is called correctly for 'single' strategy case in setStrategyCallback
  const setStrategyCallback = useCallback((newStrategy: string) => { if (newStrategy !== strategy) { const nextNumGpus = newStrategy === 'single' ? 1 : (numGpus >= 2 ? numGpus : 2); setNumGpusInternal(nextNumGpus); // Set internal state FIRST
   setStrategyInternal(newStrategy); /* useEffect handles reset */ } }, [numGpus, strategy]); // Removed reset from deps
  const setNumGpusCallback = useCallback((num: number) => { if (num !== numGpus && strategy !== 'single') { setNumGpusInternal(num); /* useEffect handles reset */ } }, [numGpus, strategy]);

  // Effect to reset simulation state ONLY when strategy or numGpus changes (after initial mount)
  const isInitialMount = useRef(true);
  useEffect(() => { if (isInitialMount.current) { isInitialMount.current = false; updateGpuStatesForStep(0); } else { reset(numGpus, strategy); } }, [strategy, numGpus, reset]); // reset dependency IS needed here

  useEffect(() => clearSimulationInterval, [clearSimulationInterval]); // Cleanup

  const value = { currentStep, totalSteps, isPlaying, strategy, numGpus, gpuStates, stepDetails, play, pause, nextStep: () => advanceStep(1), prevStep, reset: () => reset(), setStrategy: setStrategyCallback, setNumGpus: setNumGpusCallback };
  return ( <SimulationContext.Provider value={value}>{children}</SimulationContext.Provider> );
};
export const useSimulation = () => { const context = useContext(SimulationContext); if (context === undefined) throw new Error('useSimulation must be used within a SimulationProvider'); return context; };
