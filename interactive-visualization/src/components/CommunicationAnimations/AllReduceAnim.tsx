import React from 'react';
import type { Point } from '../../hooks/useGpuPositions';
import type { CommDataType } from '../../context/types';

interface AllReduceAnimProps {
  isActive: boolean;
  gpuPositions: Point[];
  centerPos: Point;
  dataType?: CommDataType;
}

export const AllReduceAnim: React.FC<AllReduceAnimProps> = (props) => {
  // Placeholder implementation
  if (!props.isActive) return null;
  console.log("Rendering AllReduceAnim (Placeholder)", props);
  return (
    <div style={{ position: 'absolute', top: 0, left: 0, zIndex: 20, color: 'green', background: 'rgba(255,255,255,0.7)' }}>
      AllReducing {props.dataType}...
    </div>
  );
}; 