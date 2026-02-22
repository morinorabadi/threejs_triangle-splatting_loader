import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { FormatLoader } from './formatLoader';
import { resolveSemantics } from './semantics/resolveSemantics';
import { createMeshRenderable } from './renderers/meshRenderer';
import { createSplatRenderable } from './renderers/splatRenderer';
import type { ParsedModel, Renderable, SemanticData, SplatParams } from './types';

interface ViewerStats {
  fps: number;
  triangles: number;
  points: number;
  lastTime: number;
  frameCount: number;
}

export class TriangleSplattingViewer {
  private canvas: HTMLCanvasElement;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private controls: OrbitControls;
  private renderable: Renderable | null;
  private modelData: ParsedModel | null;
  private semanticData: SemanticData | null;
  private clock: THREE.Clock;
  private splatSettings: SplatParams;
  private stats: ViewerStats;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera();
    this.renderer = new THREE.WebGLRenderer({ canvas });
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.renderable = null;
    this.modelData = null;
    this.semanticData = null;
    this.clock = new THREE.Clock();
    this.splatSettings = {
      alphaCurveK: 1.0,
      sigmaScale: 0.35,
      coveredAlphaFloor: 0.0,
    };

    this.stats = {
      fps: 0,
      triangles: 0,
      points: 0,
      lastTime: Date.now(),
      frameCount: 0,
    };

    this.init();
  }

  private init(): void {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x000000);

    this.camera = new THREE.PerspectiveCamera(
      45,
      window.innerWidth / window.innerHeight,
      0.1,
      1000,
    );
    this.camera.position.set(0, 0, 3);

    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
    });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.9));

    const keyLight = new THREE.DirectionalLight(0xffffff, 0.6);
    keyLight.position.set(5, 10, 7);
    this.scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0xffffff, 0.2);
    rimLight.position.set(-5, -5, -10);
    this.scene.add(rimLight);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.zoomSpeed = 1.0;
    this.controls.rotateSpeed = 0.5;
    this.controls.panSpeed = 0.5;

    window.addEventListener('resize', () => this.onWindowResize());

    const fileInput = document.getElementById('fileInput') as HTMLInputElement | null;
    fileInput?.addEventListener('change', (event) => this.loadFile(event));

    const renderMode = document.getElementById('renderMode') as HTMLSelectElement | null;
    renderMode?.addEventListener('change', () => this.applyRenderMode());

    const alphaCurve = document.getElementById('alphaCurve') as HTMLInputElement | null;
    alphaCurve?.addEventListener('input', (event) => {
      const value = Number((event.target as HTMLInputElement).value);
      this.splatSettings.alphaCurveK = value;
      const el = document.getElementById('alphaCurveValue');
      if (el) el.textContent = value.toFixed(1);
      this.applySplatSettings();
    });

    const sigmaScale = document.getElementById('sigmaScale') as HTMLInputElement | null;
    sigmaScale?.addEventListener('input', (event) => {
      const value = Number((event.target as HTMLInputElement).value);
      this.splatSettings.sigmaScale = value;
      const el = document.getElementById('sigmaScaleValue');
      if (el) el.textContent = value.toFixed(2);
      this.applySplatSettings();
    });

    const alphaFloor = document.getElementById('alphaFloor') as HTMLInputElement | null;
    alphaFloor?.addEventListener('input', (event) => {
      const value = Number((event.target as HTMLInputElement).value);
      this.splatSettings.coveredAlphaFloor = value;
      const el = document.getElementById('alphaFloorValue');
      if (el) el.textContent = value.toFixed(2);
      this.applySplatSettings();
    });

    this.animate();
  }

  private async loadFile(event: Event): Promise<void> {
    const input = event.target as HTMLInputElement;
    const file = input?.files?.[0];
    if (!file) {
      return;
    }

    this.showLoading(true, `Parsing ${file.name}...`);
    this.setStatus(`Loading ${file.name}`);

    try {
      this.modelData = await FormatLoader.loadFile(file);
      this.setStatus(`Parsed ${this.modelData.format} v${this.modelData.version}`);

      this.showLoading(true, 'Resolving tensor semantics...');
      this.semanticData = resolveSemantics(this.modelData, { opacityMode: 'auto' });
      this.applyRenderMode();

      const modelName = document.getElementById('modelName');
      const triangleCount = document.getElementById('triangleCount');
      const pointCount = document.getElementById('pointCount');
      if (modelName) modelName.textContent = file.name;
      if (triangleCount) triangleCount.textContent = String(this.stats.triangles);
      if (pointCount) pointCount.textContent = String(this.stats.points);

      this.showLoading(false);
    } catch (error) {
      console.error(error);
      this.showLoading(false);
      const message = error instanceof Error ? error.message : String(error);
      this.setStatus(`Load error: ${message}`);
      alert(`Error loading file: ${message}`);
    }
  }

  private applyRenderMode(): void {
    if (!this.semanticData) {
      return;
    }

    const modeSelect = document.getElementById('renderMode') as HTMLSelectElement | null;
    const mode = modeSelect?.value ?? 'mesh';

    this.showLoading(true, `Building ${mode} renderer...`);

    try {
      this.clearRenderable();

      let built: Renderable;
      if (mode === 'splat') {
        try {
          built = createSplatRenderable(this.semanticData);
          this.setStatus('Rendering with experimental splat mode');
        } catch (error) {
          console.warn('Splat renderer unavailable, falling back to mesh:', error);
          built = createMeshRenderable(this.semanticData);
          const message = error instanceof Error ? error.message : String(error);
          this.setStatus(`Splat unavailable (${message}); fallback to mesh`);
        }
      } else {
        built = createMeshRenderable(this.semanticData);
        this.setStatus('Rendering with mesh mode');
      }

      this.renderable = built;
      this.scene.add(built.object3d);
      this.applySplatSettings();

      this.stats.triangles = built.stats.triangles;
      this.stats.points = built.stats.points;

      this.fitCamera();
      this.showLoading(false);
    } catch (error) {
      this.showLoading(false);
      const message = error instanceof Error ? error.message : String(error);
      this.setStatus(`Render error: ${message}`);
      throw error;
    }
  }

  private applySplatSettings(): void {
    if (typeof this.renderable?.setSplatParams !== 'function') {
      return;
    }
    this.renderable.setSplatParams(this.splatSettings);
  }

  private clearRenderable(): void {
    if (!this.renderable?.object3d) {
      return;
    }

    this.scene.remove(this.renderable.object3d);

    const obj = this.renderable.object3d as THREE.Mesh;
    if (typeof this.renderable.dispose === 'function') {
      this.renderable.dispose();
    } else {
      if (obj.geometry) {
        obj.geometry.dispose();
      }

      const material = obj.material;
      if (material) {
        if (Array.isArray(material)) {
          material.forEach((mat) => mat.dispose());
        } else {
          material.dispose();
        }
      }
    }

    this.renderable = null;
  }

  private fitCamera(): void {
    if (!this.renderable?.object3d) {
      return;
    }

    const box = new THREE.Box3().setFromObject(this.renderable.object3d);
    if (box.isEmpty()) {
      return;
    }

    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const fov = this.camera.fov * (Math.PI / 180);
    let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));

    cameraZ *= 1.5;

    this.camera.position.set(center.x, center.y + size.y * 0.2, center.z + cameraZ);
    this.camera.lookAt(center);
    this.controls.target.copy(center);
    this.controls.update();
    this.camera.updateProjectionMatrix();
  }

  private animate(): void {
    requestAnimationFrame(() => this.animate());

    this.controls.update();
    this.updateStats();
    if (typeof this.renderable?.render === 'function') {
      this.renderable.render(this.renderer, this.scene, this.camera);
    } else {
      this.renderer.render(this.scene, this.camera);
    }

    const fps = document.getElementById('fps');
    const renderedTriangles = document.getElementById('renderedTriangles');
    const renderedPoints = document.getElementById('renderedPoints');

    if (fps) fps.textContent = String(Math.round(this.stats.fps));
    if (renderedTriangles) renderedTriangles.textContent = String(this.stats.triangles);
    if (renderedPoints) renderedPoints.textContent = String(this.stats.points);
  }

  private updateStats(): void {
    this.stats.frameCount++;
    const now = Date.now();
    const elapsed = now - this.stats.lastTime;

    if (elapsed >= 1000) {
      this.stats.fps = (this.stats.frameCount * 1000) / elapsed;
      this.stats.frameCount = 0;
      this.stats.lastTime = now;
    }
  }

  private onWindowResize(): void {
    const width = window.innerWidth;
    const height = window.innerHeight;

    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  private showLoading(show: boolean, text = 'Loading...'): void {
    const loading = document.getElementById('loading');
    const status = document.getElementById('loadingStatus');

    if (loading) loading.style.display = show ? 'block' : 'none';
    if (status) status.textContent = text;
  }

  private setStatus(text: string): void {
    const el = document.getElementById('statusText');
    if (el) {
      el.textContent = text;
    }
  }
}

declare global {
  interface Window {
    viewer?: TriangleSplattingViewer;
  }
}

function initViewer(): void {
  const existingCanvas = document.querySelector('canvas') as HTMLCanvasElement | null;
  const canvas =
    existingCanvas ??
    (() => {
      const created = document.createElement('canvas');
      created.id = 'canvas';
      document.body.appendChild(created);
      return created;
    })();

  window.viewer = new TriangleSplattingViewer(canvas);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initViewer);
} else {
  initViewer();
}
