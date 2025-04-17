import { useState, useEffect, RefObject } from 'react';

export interface Point { x: number; y: number; }
export interface GpuPositionInfo {
    id: number;
    node: HTMLElement; // The actual DOM node
    center: Point;
    bounds: DOMRect; // Full bounding box
}

/**
 * Custom hook to calculate and update the screen positions
 * of GPU components within a container.
 * @param gpuContainerRef Ref to the container holding the GPU elements.
 * @param numGpus Expected number of GPUs (triggers recalculation).
 * @param dependency Trigger recalculation when this changes (e.g., currentStep).
 */
export function useGpuPositions(
    gpuContainerRef: RefObject<HTMLDivElement>,
    numGpus: number,
    dependency: any // Recalculate when this dependency changes
): { positions: GpuPositionInfo[], containerRect: DOMRect | null } {
    const [positions, setPositions] = useState<GpuPositionInfo[]>([]);
    const [containerRect, setContainerRect] = useState<DOMRect | null>(null);

    useEffect(() => {
        const calculatePositions = () => {
            const container = gpuContainerRef.current;
            if (!container || numGpus <= 0) {
                setPositions([]);
                setContainerRect(null);
                return;
            }

            const newPositions: GpuPositionInfo[] = [];
            const gpuElements = container.querySelectorAll('[data-gpu-id]'); // Use data attribute selector
            const contRect = container.getBoundingClientRect();
            setContainerRect(contRect); // Store container bounds

             // Ensure we only process the expected number, even if more are rendered temporarily
            const elementsToProcess = Array.from(gpuElements).slice(0, numGpus);

            elementsToProcess.forEach((el) => {
                const htmlEl = el as HTMLElement;
                const idStr = htmlEl.dataset.gpuId; // Get ID from data attribute
                if (idStr === undefined) return; // Skip if no ID

                const id = parseInt(idStr, 10);
                const rect = htmlEl.getBoundingClientRect();
                const centerX = rect.left + rect.width / 2 - contRect.left; // Relative X center
                const centerY = rect.top + rect.height / 2 - contRect.top; // Relative Y center

                newPositions[id] = { // Use ID as index
                    id: id,
                    node: htmlEl,
                    center: { x: centerX, y: centerY },
                    bounds: rect // Store full bounds relative to viewport (useful later)
                };
            });

             // Fill any gaps if indexing skipped some (shouldn't happen with querySelectorAll)
             const finalPositions = Array.from({ length: numGpus }, (_, i) => newPositions[i] || null).filter(p => p !== null) as GpuPositionInfo[];

            // Only update if the positions have actually changed significantly
            setPositions(prevPos => {
                 if (finalPositions.length !== prevPos.length ||
                     finalPositions.some((p, i) => !prevPos[i] || Math.abs(p.center.x - prevPos[i].center.x) > 1 || Math.abs(p.center.y - prevPos[i].center.y) > 1)) {
                    // console.log("Updating GPU positions", finalPositions);
                    return finalPositions;
                 }
                 return prevPos; // No significant change
            });
        };

        // Initial calculation
        const timerId = setTimeout(calculatePositions, 50); // Slight delay for render finalization

        // Recalculate on resize using ResizeObserver for better performance
        const resizeObserver = new ResizeObserver(calculatePositions);
        if (gpuContainerRef.current) {
            resizeObserver.observe(gpuContainerRef.current);
        }

        // Cleanup
        return () => {
            clearTimeout(timerId);
            resizeObserver.disconnect();
        };
    // Dependencies: Recalculate if container ref, numGpus, or the external dependency changes
    }, [gpuContainerRef, numGpus, dependency]);

    return { positions, containerRect };
} 