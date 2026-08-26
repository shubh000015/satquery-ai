"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";

export function EarthCanvas() {
  const wrap = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const host = wrap.current;
    if (!host) return;

    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: "high-performance",
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x05070f, 1);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";
    renderer.domElement.style.display = "block";
    host.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x05070f, 0.012);
    const camera = new THREE.PerspectiveCamera(36, 1, 0.1, 400);
    camera.position.set(-0.35, 0.08, 3.55);

    const loader = new THREE.TextureLoader();
    const load = (url: string) =>
      new Promise<THREE.Texture>((resolve, reject) => {
        loader.load(
          url,
          (t) => {
            t.colorSpace = THREE.SRGBColorSpace;
            t.anisotropy = 8;
            resolve(t);
          },
          undefined,
          reject
        );
      });

    let raf = 0;
    let disposed = false;
    let visible = true;
    const mouse = { x: 0, y: 0 };
    const target = { x: 0, y: 0 };
    const t0 = performance.now();
    const disposables: THREE.Object3D[] = [];

    const onMove = (e: MouseEvent) => {
      target.x = (e.clientX / window.innerWidth) * 2 - 1;
      target.y = (e.clientY / window.innerHeight) * 2 - 1;
    };
    window.addEventListener("mousemove", onMove);

    const resize = () => {
      const w = host.clientWidth;
      const h = host.clientHeight;
      renderer.setSize(w, h, false);
      camera.aspect = w / Math.max(h, 1);
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(host);
    resize();

    let startTick: (() => void) | null = null;
    const io = new IntersectionObserver(
      ([entry]) => {
        visible = entry.isIntersecting;
        if (visible && !disposed) startTick?.();
      },
      { threshold: 0.08 }
    );
    io.observe(host);

    const stars = makeStars(3600);
    scene.add(stars);
    disposables.push(stars);

    const dust = makeDust();
    scene.add(dust);
    disposables.push(dust);

    const sun = new THREE.Vector3(-3.5, 1.2, 2.6).normalize();

    Promise.all([
      load("/space/earth.jpg"),
      load("/space/night.jpg"),
      load("/space/clouds.png"),
      load("/space/earth-spec.jpg"),
    ])
      .then(([day, night, clouds, spec]) => {
        if (disposed) return;

        const earth = new THREE.Mesh(
          new THREE.SphereGeometry(1, 128, 96),
          new THREE.ShaderMaterial({
            uniforms: {
              dayMap: { value: day },
              nightMap: { value: night },
              specMap: { value: spec },
              sunDirection: { value: sun },
            },
            vertexShader: `
              varying vec2 vUv;
              varying vec3 vWorldNormal;
              void main() {
                vUv = uv;
                vWorldNormal = normalize(mat3(modelMatrix) * normal);
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
              }
            `,
            fragmentShader: `
              uniform sampler2D dayMap;
              uniform sampler2D nightMap;
              uniform sampler2D specMap;
              uniform vec3 sunDirection;
              varying vec2 vUv;
              varying vec3 vWorldNormal;
              void main() {
                vec3 n = normalize(vWorldNormal);
                float ndl = dot(n, sunDirection);
                float f = smoothstep(-0.08, 0.32, ndl);
                vec3 dayC = texture2D(dayMap, vUv).rgb;
                vec3 nightC = texture2D(nightMap, vUv).rgb * vec3(1.3, 0.95, 0.62) * 1.7;
                vec3 color = mix(nightC, dayC, f);
                float ocean = texture2D(specMap, vUv).r;
                float spec = pow(max(ndl, 0.0), 22.0) * ocean * 0.55;
                color += vec3(0.85, 0.9, 1.0) * spec;
                float rim = pow(1.0 - max(ndl, 0.0), 3.0) * 0.12;
                color += vec3(1.0, 0.5, 0.28) * rim;
                gl_FragColor = vec4(color, 1.0);
              }
            `,
          })
        );

        const cloudMesh = new THREE.Mesh(
          new THREE.SphereGeometry(1.014, 128, 96),
          new THREE.MeshBasicMaterial({
            map: clouds,
            transparent: true,
            opacity: 0.5,
            depthWrite: false,
          })
        );

        const atmosphere = new THREE.Mesh(
          new THREE.SphereGeometry(1.11, 96, 64),
          new THREE.ShaderMaterial({
            uniforms: { sunDirection: { value: sun } },
            vertexShader: `
              varying vec3 vNormal;
              varying vec3 vView;
              varying vec3 vWorldNormal;
              void main() {
                vec4 mv = modelViewMatrix * vec4(position, 1.0);
                vNormal = normalize(normalMatrix * normal);
                vView = normalize(-mv.xyz);
                vWorldNormal = normalize(mat3(modelMatrix) * normal);
                gl_Position = projectionMatrix * mv;
              }
            `,
            fragmentShader: `
              uniform vec3 sunDirection;
              varying vec3 vNormal;
              varying vec3 vView;
              varying vec3 vWorldNormal;
              void main() {
                float f = pow(1.0 - abs(dot(normalize(vNormal), normalize(vView))), 2.6);
                float lit = smoothstep(-0.1, 0.6, dot(normalize(vWorldNormal), sunDirection));
                vec3 dawn = vec3(1.0, 0.52, 0.28);
                vec3 sky = vec3(0.32, 0.55, 1.0);
                vec3 col = mix(dawn, sky, lit);
                gl_FragColor = vec4(col, f * 0.95);
              }
            `,
            blending: THREE.AdditiveBlending,
            side: THREE.BackSide,
            transparent: true,
            depthWrite: false,
          })
        );

        const group = new THREE.Group();
        group.rotation.z = (23.4 * Math.PI) / 180;
        earth.rotation.y = 1.18;
        group.add(earth, cloudMesh, atmosphere);
        group.position.set(1.08, -0.22, 0);
        group.scale.setScalar(1.52);
        scene.add(group);
        disposables.push(group);

        const orbitRing = makeOrbitRing();
        orbitRing.position.copy(group.position);
        orbitRing.rotation.set(0.9, 0.3, 0);
        scene.add(orbitRing);
        disposables.push(orbitRing);

        const sat = makeSatellite();
        scene.add(sat);
        disposables.push(sat);

        scene.add(new THREE.AmbientLight(0x38445e, 0.35));
        const key = new THREE.DirectionalLight(0xfff1dd, 2.2);
        key.position.copy(sun).multiplyScalar(8);
        scene.add(key);
        const rim = new THREE.DirectionalLight(0xff9455, 0.5);
        rim.position.set(4, 0.8, -3);
        scene.add(rim);

        const tick = () => {
          if (disposed) return;
          if (!visible) return;
          const t = (performance.now() - t0) / 1000;
          mouse.x += (target.x - mouse.x) * 0.03;
          mouse.y += (target.y - mouse.y) * 0.03;

          if (!reduce) {
            earth.rotation.y += 0.00055;
            cloudMesh.rotation.y += 0.00078;
            const a = t * 0.12;
            const orbitR = 2.85;
            sat.position.set(
              group.position.x + Math.cos(a) * orbitR * 0.72,
              group.position.y + Math.sin(a * 1.6) * 0.32 + 0.35,
              Math.sin(a) * orbitR * 0.85
            );
            sat.rotation.y += 0.008;
            orbitRing.rotation.z += 0.0003;
            stars.rotation.y += 0.00005;
          }

          camera.position.x = -0.35 + mouse.x * 0.22;
          camera.position.y = 0.08 - mouse.y * 0.12;
          camera.position.z = 3.55;
          camera.lookAt(0.62, -0.12, 0);
          renderer.render(scene, camera);
          raf = window.requestAnimationFrame(tick);
        };
        startTick = () => {
          window.cancelAnimationFrame(raf);
          tick();
        };
        tick();
      })
      .catch(() => {
        if (disposed) return;
        const fb = new THREE.Mesh(
          new THREE.SphereGeometry(1, 64, 48),
          new THREE.MeshStandardMaterial({ color: 0x1a3a6c, roughness: 0.8 })
        );
        fb.position.set(1.08, -0.22, 0);
        fb.scale.setScalar(1.52);
        scene.add(fb);
        const tick = () => {
          if (disposed) return;
          fb.rotation.y += 0.001;
          renderer.render(scene, camera);
          raf = window.requestAnimationFrame(tick);
        };
        tick();
      });

    return () => {
      disposed = true;
      window.cancelAnimationFrame(raf);
      window.removeEventListener("mousemove", onMove);
      ro.disconnect();
      io.disconnect();
      disposables.forEach((o) => scene.remove(o));
      scene.traverse((obj) => {
        const mesh = obj as THREE.Mesh;
        mesh.geometry?.dispose?.();
        const mat = mesh.material as THREE.Material | THREE.Material[] | undefined;
        if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
        else mat?.dispose?.();
      });
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  return <div ref={wrap} className="absolute inset-0" />;
}

function makeStars(count: number) {
  const pos = new Float32Array(count * 3);
  const col = new Float32Array(count * 3);
  const sz = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const r = 20 + Math.random() * 80;
    const u = Math.random();
    const v = Math.random();
    const theta = 2 * Math.PI * u;
    const phi = Math.acos(2 * v - 1);
    pos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
    pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
    pos[i * 3 + 2] = r * Math.cos(phi);
    const warm = Math.random() > 0.75;
    const c = 0.6 + Math.random() * 0.4;
    col[i * 3] = warm ? 1 : c;
    col[i * 3 + 1] = warm ? 0.78 : c;
    col[i * 3 + 2] = warm ? 0.55 : 1;
    sz[i] = 0.02 + Math.random() * 0.06;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  return new THREE.Points(
    g,
    new THREE.PointsMaterial({
      size: 0.05,
      vertexColors: true,
      transparent: true,
      opacity: 0.95,
      depthWrite: false,
      sizeAttenuation: true,
    })
  );
}

function makeDust() {
  const count = 340;
  const pos = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    pos[i * 3] = (Math.random() - 0.5) * 12;
    pos[i * 3 + 1] = (Math.random() - 0.5) * 6;
    pos[i * 3 + 2] = -2 - Math.random() * 6;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  return new THREE.Points(
    g,
    new THREE.PointsMaterial({
      size: 0.12,
      color: 0xff9a70,
      transparent: true,
      opacity: 0.1,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    })
  );
}

function makeOrbitRing() {
  const seg = 256;
  const r = 2.35;
  const pts: THREE.Vector3[] = [];
  for (let i = 0; i <= seg; i++) {
    const a = (i / seg) * Math.PI * 2;
    pts.push(new THREE.Vector3(Math.cos(a) * r, 0, Math.sin(a) * r));
  }
  const g = new THREE.BufferGeometry().setFromPoints(pts);
  const m = new THREE.LineBasicMaterial({
    color: 0xf2ead4,
    transparent: true,
    opacity: 0.22,
  });
  return new THREE.Line(g, m);
}

function makeSatellite() {
  const g = new THREE.Group();
  const body = new THREE.MeshStandardMaterial({
    color: 0xdad1b8,
    metalness: 0.6,
    roughness: 0.35,
  });
  const panelMat = new THREE.MeshStandardMaterial({
    color: 0x1a2a52,
    metalness: 0.7,
    roughness: 0.2,
    emissive: 0x0a1a3a,
    emissiveIntensity: 0.5,
  });
  const trim = new THREE.MeshStandardMaterial({
    color: 0xff7a45,
    metalness: 0.5,
    roughness: 0.4,
    emissive: 0xff7a45,
    emissiveIntensity: 0.15,
  });
  const bus = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.06, 0.06), body);
  const dish = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.02, 0.02, 12), trim);
  dish.rotation.x = Math.PI / 2;
  dish.position.z = 0.045;
  const p1 = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.003, 0.08), panelMat);
  const p2 = p1.clone();
  p1.position.x = 0.15;
  p2.position.x = -0.15;
  g.add(bus, dish, p1, p2);
  g.scale.setScalar(1.2);
  return g;
}
