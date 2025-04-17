import React from 'react';
import type { Point } from '../../hooks/useGpuPositions';
import type { CommDataType } from '../../context/types';

interface AllGatherAnimProps {
  isActive: boolean;
  gpuPositions: Point[];
  dataType?: CommDataType;
}

export const AllGatherAnim: React.FC<AllGatherAnimProps> = (props) => {
  // Placeholder implementation
  if (!props.isActive) return null;
  console.log("Rendering AllGatherAnim (Placeholder)", props);
  return (
    <div style={{ position: 'absolute', top: 0, left: 0, zIndex: 20, color: 'purple', background: 'rgba(255,255,255,0.7)' }}>
      AllGathering {props.dataType}...
    </div>
  );
}; 