/**
 * WebGL Shaders for Triangle Splatting
 */

export const SHADERS = {
  splat_triangle_vs: `
        attribute vec3 color;
        attribute float alpha;
        attribute float splatSigma;
        attribute vec3 barycentric;

        varying vec3 vColor;
        varying float vAlpha;
        varying float vSigma;
        varying vec3 vBarycentric;

        void main() {
            vColor = color;
            vAlpha = alpha;
            vSigma = splatSigma;
            vBarycentric = barycentric;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
    `,

  splat_triangle_fs: `
        precision highp float;

        varying vec3 vColor;
        varying float vAlpha;
        varying float vSigma;
        varying vec3 vBarycentric;
        uniform float uSigmaScale;
        uniform float uEdgeSoftness;
        uniform float uAlphaCurveK;
        uniform float uCoveredAlphaFloor;

        void main() {
            float edge = min(min(vBarycentric.x, vBarycentric.y), vBarycentric.z);
            float grad = max(fwidth(edge), 1e-5);
            float edgePx = edge / grad;

            // Prevent very large sigma values from washing objects out.
            float sigmaPx = clamp(vSigma * uSigmaScale, 1e-3, 0.35);
            float window = 1.0 - exp(-(edgePx * edgePx) / (2.0 * sigmaPx * sigmaPx));
            window = pow(clamp(window, 0.0, 1.0), uEdgeSoftness);

            float baseAlpha = clamp(vAlpha * window, 0.0, 1.0);

            float effectiveK = 1.0 + (uAlphaCurveK - 1.0) * 0.25;
            float outAlpha = 1.0 - pow(1.0 - baseAlpha, effectiveK);
            outAlpha = clamp(outAlpha, 0.0, 1.0);
            float covered = smoothstep(0.10, 0.40, window);

            outAlpha = max(outAlpha, (uCoveredAlphaFloor * 0.5) * covered);
            if (outAlpha <= 1e-3) discard;

            gl_FragColor = vec4(vColor * outAlpha, outAlpha);
        }
    `,

  triangle_vs: `
        #define MAX_SH_DEGREE 3

        uniform mat4 projectionMatrix;
        uniform mat4 viewMatrix;
        uniform mat4 modelMatrix;
        uniform vec3 cameraPos;

        // SH coefficient arrays
        uniform sampler2D uShCoefficients;
        uniform int sh_degree;

        attribute vec3 position;
        attribute vec3 color_dc;
        attribute float opacity;
        attribute int sh_index;

        varying vec3 vColor;
        varying float vOpacity;

        vec3 evalSH(vec3 dir, int sh_degree) {
            vec3 color = vec3(0.0);
            color = color_dc;
            return color;
        }

        void main() {
            gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(position, 1.0);
            vColor = color_dc;
            vOpacity = opacity;
        }
    `,

  triangle_fs: `
        precision mediump float;

        varying vec3 vColor;
        varying float vOpacity;

        void main() {
            gl_FragColor = vec4(vColor, vOpacity);
        }
    `,

  points_vs: `
        uniform mat4 projectionMatrix;
        uniform mat4 viewMatrix;
        uniform mat4 modelMatrix;
        uniform float pointSize;

        attribute vec3 position;
        attribute vec3 color;
        attribute float opacity;

        varying vec3 vColor;
        varying float vOpacity;

        void main() {
            gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(position, 1.0);
            gl_PointSize = pointSize;

            vColor = color;
            vOpacity = opacity;
        }
    `,

  points_fs: `
        precision mediump float;

        varying vec3 vColor;
        varying float vOpacity;

        void main() {
            vec2 circCoord = 2.0 * gl_PointCoord - 1.0;
            if (dot(circCoord, circCoord) > 1.0) discard;

            gl_FragColor = vec4(vColor, vOpacity);
        }
    `,

  evaluate_sh: `
        precision highp float;

        vec4 evaluateSHBasis0(vec3 dir) {
            vec4 sh = vec4(0.282095, 0.488603 * dir.y, 0.488603 * dir.z, 0.488603 * dir.x);
            return sh;
        }

        vec4 evaluateSHBasis1(vec3 dir) {
            vec4 sh = vec4(
                1.092548 * dir.x * dir.y,
                1.092548 * dir.y * dir.z,
                0.315392 * (3.0 * dir.z * dir.z - 1.0),
                1.092548 * dir.x * dir.z
            );
            return sh;
        }

        vec4 evaluateSHBasis2(vec3 dir) {
            vec4 sh = vec4(
                0.546274 * (5.0 * dir.z * dir.z - 1.0) * dir.x,
                2.429641 * dir.x * dir.y * dir.z,
                0.429043 * (5.0 * dir.z * dir.z - 3.0),
                2.429641 * dir.y * dir.z
            );
            return sh;
        }

        vec3 computeColorFromSH(
            vec3 sh_coeff_dc,
            vec4 sh_coeff_l1_1,
            vec4 sh_coeff_l1_2,
            vec4 sh_coeff_l1_3,
            vec3 dir
        ) {
            dir = normalize(dir);

            vec3 color = 0.5 * sh_coeff_dc;

            if (length(sh_coeff_l1_1) > 0.0 || length(sh_coeff_l1_2) > 0.0 || length(sh_coeff_l1_3) > 0.0) {
                vec4 sh_basis = evaluateSHBasis0(dir);
                color += 0.5 * sh_coeff_l1_1.x * sh_basis.x;
                color += 0.5 * sh_coeff_l1_1.y * sh_basis.y;
                color += 0.5 * sh_coeff_l1_1.z * sh_basis.z;
                color += 0.5 * sh_coeff_l1_1.w * sh_basis.w;
            }

            return color;
        }
    `,
} as const;

export function compileShader(
  gl: WebGLRenderingContext,
  source: string,
  type: number,
): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) {
    return null;
  }

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error('Shader compilation error:', gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }

  return shader;
}

export function createShaderProgram(
  gl: WebGLRenderingContext,
  vertexSource: string,
  fragmentSource: string,
): WebGLProgram | null {
  const vs = compileShader(gl, vertexSource, gl.VERTEX_SHADER);
  const fs = compileShader(gl, fragmentSource, gl.FRAGMENT_SHADER);

  if (!vs || !fs) return null;

  const program = gl.createProgram();
  if (!program) {
    return null;
  }

  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error('Program linking error:', gl.getProgramInfoLog(program));
    return null;
  }

  gl.deleteShader(vs);
  gl.deleteShader(fs);

  return program;
}
