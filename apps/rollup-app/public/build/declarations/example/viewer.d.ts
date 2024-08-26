import { PerspectiveCamera, Scene } from 'three';
import { PointCloudOctree } from '../pointcloud';
export declare class Viewer {
    /**
     * The element where we will insert our canvas.
     */
    private targetEl;
    /**
     * The ThreeJS renderer used to render the scene.
     */
    private renderer;
    /**
     * Our scene which will contain the point cloud.
     */
    scene: Scene;
    /**
     * The camera used to view the scene.
     */
    camera: PerspectiveCamera;
    /**
     * Controls which update the position of the camera.
     */
    cameraControls: any;
    /**
     * Out potree instance which handles updating point clouds, keeps track of loaded nodes, etc.
     */
    private potree_v2;
    /**
     * Array of point clouds which are in the scene and need to be updated.
     */
    private pointClouds;
    /**
     * The time (milliseconds) when `loop()` was last called.
     */
    private prevTime;
    /**
     * requestAnimationFrame handle we can use to cancel the viewer loop.
     */
    private reqAnimationFrameHandle;
    /**
     * Initializes the viewer into the specified element.
     *
     * @param targetEl
     *    The element into which we should add the canvas where we will render the scene.
     */
    initialize(targetEl: HTMLElement): void;
    /**
     * Performs any cleanup necessary to destroy/remove the viewer from the page.
     */
    destroy(): void;
    /**
     * Loads a point cloud into the viewer and returns it.
     *
     * @param fileName
     *    The name of the point cloud which is to be loaded.
     * @param baseUrl
     *    The url where the point cloud is located and from where we should load the octree nodes.
     */
    load(fileName: string, baseUrl: string): Promise<PointCloudOctree>;
    add(pco: PointCloudOctree): void;
    disposePointCloud(pointCloud: PointCloudOctree): void;
    /**
     * Updates the point clouds, cameras or any other objects which are in the scene.
     *
     * @param dt
     *    The time, in milliseconds, since the last update.
     */
    update(_: number): void;
    /**
     * Renders the scene into the canvas.
     */
    render(): void;
    /**
     * The main loop of the viewer, called at 60FPS, if possible.
     */
    loop: (time: number) => void;
    /**
     * Triggered anytime the window gets resized.
     */
    resize: () => void;
}
