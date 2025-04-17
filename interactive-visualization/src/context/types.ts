export interface GpuState { id: number; paramMemory: number; activationMemory: number; gradientMemory: number; optStateMemory: number; status: 'idle' | 'computing' | 'communicating'; currentLayerName?: string; isParamsTempFull?: boolean; dataShardId?: number; }
export type CommOperation = 'AllReduce' | 'P2P' | 'AllGather' | 'ReduceScatter' | 'AlltoAll' | string;
export type CommDataType = 'Activations' | 'Gradients' | 'Params' | 'Tokens' | 'KV' | string;
export interface StepDetail {
    step: number;
    type?: string;
    layer?: string;
    subStep?: string;
    parallel?: boolean;
    direction?: 'forward' | 'backward';
    operation?: CommOperation | string;
    dataType?: CommDataType;
    description: string;
    notation?: string;
    strategy?: string;
    // Fields for activation memory tracking
    activationProduced?: string | null; // Name of the layer whose activation is produced by this step
    activationConsumedLayer?: string | null; // Name of the layer whose activation is consumed by this backward step

    // Fields for TP Visualization
    tpExecutionType?: string; // e.g., 'ColumnParallel', 'RowParallel', 'LocalAttention', 'Replicated'
    inputDesc?: string; // For TpLayerExecutionViz
    weightDesc?: string; // For TpLayerExecutionViz
    outputDesc?: string; // For TpLayerExecutionViz
}
