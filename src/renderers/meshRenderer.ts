import * as THREE from 'three';
import type { Renderable, SemanticData } from '../types';

export function createMeshRenderable(semanticData: SemanticData): Renderable {
  const { positions, colorDC, opacity, stats } = semanticData;

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions.data, 3));

  if (colorDC?.data) {
    geometry.setAttribute('color', new THREE.BufferAttribute(colorDC.data, 3));
  }

  if (opacity?.data) {
    geometry.setAttribute('alpha', new THREE.BufferAttribute(opacity.data, 1));
  }

  const indexCount = stats.totalVertices;
  const indexArray =
    indexCount > 65535 ? new Uint32Array(indexCount) : new Uint16Array(indexCount);
  for (let i = 0; i < indexCount; i++) {
    indexArray[i] = i;
  }
  geometry.setIndex(new THREE.BufferAttribute(indexArray, 1));

  const material = new THREE.MeshPhongMaterial({
    color: 0xffffff,
    vertexColors: Boolean(colorDC?.data),
    side: THREE.DoubleSide,
    flatShading: false,
    transparent: Boolean(opacity?.data),
    opacity: 1,
    shininess: 60,
  });

  const mesh = new THREE.Mesh(geometry, material);
  return {
    object3d: mesh,
    mode: 'mesh',
    stats: {
      triangles: stats.triangleCount,
      points: stats.totalVertices,
    },
  };
}
