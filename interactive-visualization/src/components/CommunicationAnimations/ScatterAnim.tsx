import React from 'react';
import type { Point } from '../../hooks/useGpuPositions';
import type { CommDataType } from '../../context/types';

interface ScatterAnimProps {
  isActive: boolean;
  sourcePos: Point;
  targetPositions: Point[];
  dataType?: CommDataType;
}

export const ScatterAnim: React.FC<ScatterAnimProps> = (props) => {
  // Placeholder implementation
  if (!props.isActive) return null;
  console.log("Rendering ScatterAnim (Placeholder)", props);
  return (
    <div style={{ position: 'absolute', top: 0, left: 0, zIndex: 20, color: 'blue', background: 'rgba(255,255,255,0.7)' }}>
      Scattering {props.dataType}...
    </div>
  );
}; 