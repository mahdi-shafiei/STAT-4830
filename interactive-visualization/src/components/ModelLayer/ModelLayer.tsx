import React, { forwardRef } from 'react';
import styles from './ModelLayer.module.css';

interface ModelLayerProps {
  type: string;
  isHighlighted?: boolean; // Add prop for highlighting
}

// Use forwardRef to allow parent component to get a ref to the DOM element
const ModelLayer = forwardRef<HTMLDivElement, ModelLayerProps>(
    ({ type, isHighlighted }, ref) => {
        return (
            // Combine classes using template literal
            <div
                ref={ref} // Attach the forwarded ref here
                className={`${styles.modelLayer} ${isHighlighted ? styles.highlighted : ''}`}
            >
            {type}
            </div>
        );
    }
);
// Add display name for React DevTools
ModelLayer.displayName = 'ModelLayer';

export default ModelLayer;
