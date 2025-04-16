// Re-usable type definitions for the simulation context
export interface GpuState { id: number; paramMemory: number; activationMemory: number; gradientMemory: number; optStateMemory: number; status: 'idle' | 'computing' | 'communicating'; currentLayerName?: string; isParamsTempFull?: boolean; dataShardId?: number; }
export type CommOperation = 'AllReduce' | 'P2P' | 'AllGather' | 'ReduceScatter' | 'AlltoAll' | string;
export type CommDataType = 'Activations' | 'Gradients' | 'Params' | 'Tokens' | 'KV' | string;
export interface StepDetail { step: number; type?: string; layer?: string; parallel?: boolean; direction?: 'forward' | 'backward'; operation?: CommOperation | string; dataType?: CommDataType; description: string; notation?: string; strategy?: string; }
