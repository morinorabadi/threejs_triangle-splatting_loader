import * as THREE from 'three';
import { SHADERS } from '../shaders';
import type { Renderable, SemanticData, SplatParams } from '../types';

function makeCompositeMaterial(accumTexture: THREE.Texture | null): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: {
      uAccumTex: { value: accumTexture },
    },
    vertexShader: `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = vec4(position.xy, 0.0, 1.0);
      }
    `,
    fragmentShader: `
      precision highp float;
      varying vec2 vUv;
      uniform sampler2D uAccumTex;

      void main() {
        vec4 accum = texture2D(uAccumTex, vUv);
        float a = max(accum.a, 1e-6);
        vec3 color = accum.rgb / a;
        float alpha = 1.0 - exp(-accum.a);
        gl_FragColor = vec4(clamp(color, 0.0, 1.0), clamp(alpha, 0.0, 1.0));
      }
    `,
    transparent: false,
    depthTest: false,
    depthWrite: false,
  });
}

export function createSplatRenderable(semanticData: SemanticData): Renderable {
  const { positions, colorDC, opacity, sigma, stats } = semanticData;

  if (!colorDC?.data) {
    throw new Error('Splat mode requires DC color tensor');
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions.data, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colorDC.data, 3));

  const alphaData = opacity?.data ?? new Float32Array(stats.totalVertices).fill(1);
  geometry.setAttribute('alpha', new THREE.BufferAttribute(alphaData, 1));

  const sigmaData = sigma?.data ?? new Float32Array(stats.totalVertices).fill(0.1);
  geometry.setAttribute('splatSigma', new THREE.BufferAttribute(sigmaData, 1));

  const barycentric = new Float32Array(stats.totalVertices * 3);
  for (let tri = 0; tri < stats.triangleCount; tri++) {
    const base = tri * 9;
    barycentric[base] = 1;
    barycentric[base + 4] = 1;
    barycentric[base + 8] = 1;
  }
  geometry.setAttribute('barycentric', new THREE.BufferAttribute(barycentric, 3));

  const indexCount = stats.totalVertices;
  const indexArray =
    indexCount > 65535 ? new Uint32Array(indexCount) : new Uint16Array(indexCount);
  for (let i = 0; i < indexCount; i++) {
    indexArray[i] = i;
  }
  geometry.setIndex(new THREE.BufferAttribute(indexArray, 1));

  const material = new THREE.ShaderMaterial({
    vertexShader: SHADERS.splat_triangle_vs,
    fragmentShader: SHADERS.splat_triangle_fs,
    uniforms: {
      uSigmaScale: { value: 0.35 },
      uEdgeSoftness: { value: 0.9 },
      uAlphaCurveK: { value: 1.0 },
      uCoveredAlphaFloor: { value: 0.0 },
    },
    side: THREE.DoubleSide,
    transparent: true,
    premultipliedAlpha: true,
    depthWrite: false,
    depthTest: true,
    blending: THREE.AdditiveBlending,
  });

  const mesh = new THREE.Mesh(geometry, material);

  const compositeScene = new THREE.Scene();
  const compositeCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const compositeQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), makeCompositeMaterial(null));
  compositeScene.add(compositeQuad);

  let accumTarget: THREE.WebGLRenderTarget | null = null;

  function ensureTargets(renderer: THREE.WebGLRenderer): void {
    const size = renderer.getSize(new THREE.Vector2());
    const width = Math.max(1, Math.floor(size.x));
    const height = Math.max(1, Math.floor(size.y));

    if (accumTarget && accumTarget.width === width && accumTarget.height === height) {
      return;
    }

    if (accumTarget) {
      accumTarget.dispose();
    }

    accumTarget = new THREE.WebGLRenderTarget(width, height, {
      format: THREE.RGBAFormat,
      type: THREE.HalfFloatType,
      depthBuffer: true,
      stencilBuffer: false,
    });

    (compositeQuad.material as THREE.ShaderMaterial).uniforms.uAccumTex.value = accumTarget.texture;
  }

  function render(renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.Camera): void {
    ensureTargets(renderer);
    if (!accumTarget) return;

    const prevTarget = renderer.getRenderTarget();
    const prevAutoClear = renderer.autoClear;

    renderer.autoClear = true;

    renderer.setRenderTarget(accumTarget);
    renderer.setClearColor(0x000000, 0);
    renderer.clear(true, true, false);
    renderer.render(scene, camera);

    renderer.setRenderTarget(null);
    renderer.setClearColor(0x000000, 1);
    renderer.clear(true, true, false);
    renderer.render(compositeScene, compositeCamera);

    renderer.setRenderTarget(prevTarget);
    renderer.autoClear = prevAutoClear;
  }

  function dispose(): void {
    geometry.dispose();
    material.dispose();
    compositeQuad.geometry.dispose();
    (compositeQuad.material as THREE.Material).dispose();
    if (accumTarget) {
      accumTarget.dispose();
      accumTarget = null;
    }
  }

  function setSplatParams(params: Partial<SplatParams> = {}): void {
    if (typeof params.alphaCurveK === 'number') {
      material.uniforms.uAlphaCurveK.value = Math.max(1.0, Math.min(1000.0, params.alphaCurveK));
    }
    if (typeof params.sigmaScale === 'number') {
      material.uniforms.uSigmaScale.value = Math.max(0.01, Math.min(2.0, params.sigmaScale));
    }
    if (typeof params.coveredAlphaFloor === 'number') {
      material.uniforms.uCoveredAlphaFloor.value = Math.max(
        0.0,
        Math.min(0.8, params.coveredAlphaFloor),
      );
    }
  }

  return {
    object3d: mesh,
    mode: 'splat',
    stats: {
      triangles: stats.triangleCount,
      points: stats.totalVertices,
    },
    render,
    dispose,
    setSplatParams,
  };
}
