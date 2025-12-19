(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const container = $("gameContainer");
  const scoreLeftEl = $("scoreLeft");
  const scoreRightEl = $("scoreRight");
  const overlayEl = $("overlay");
  const hintEl = $("hint");

  // ─────────────────────────────────────────────────────────────
  // SETTINGS
  // ─────────────────────────────────────────────────────────────
  const SETTINGS = {
    winScore: 11,
    court: {
      width: 20,   // X-axis (horizontal)
      height: 12,  // Y-axis (vertical)
      depth: 40,   // Z-axis (towards opponent)
    },
    paddle: {
      width: 4,
      height: 3,
      depth: 0.5,
      speed: 18,
      inset: 1,
    },
    ball: {
      radius: 0.4,
      serveSpeed: 18,
      maxSpeed: 35,
      speedUpPerHit: 1.2,
    },
    ai: {
      maxSpeed: 14,
      reaction: 0.08,
      deadZone: 0.3,
    },
  };

  // ─────────────────────────────────────────────────────────────
  // STATE
  // ─────────────────────────────────────────────────────────────
  const Keys = { up: false, down: false, left: false, right: false };

  const State = {
    running: false,
    paused: false,
    gameOver: false,
    lastPointBy: 0,
    scoreL: 0,
    scoreR: 0,
    left: { x: 0, y: 0, z: 0, vx: 0, vy: 0 },
    right: { x: 0, y: 0, z: 0, vx: 0, vy: 0, targetX: 0, targetY: 0 },
    ball: { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, speed: 0 },
    lastT: 0,
    acc: 0,
  };

  // ─────────────────────────────────────────────────────────────
  // THREE.JS SETUP
  // ─────────────────────────────────────────────────────────────
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x050510);
  scene.fog = new THREE.Fog(0x050510, 30, 60);

  const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 100);
  camera.position.set(0, 8, -SETTINGS.court.depth / 2 - 10);
  camera.lookAt(0, 0, 10);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);

  // Lighting
  const ambientLight = new THREE.AmbientLight(0x222244, 0.5);
  scene.add(ambientLight);

  const mainLight = new THREE.DirectionalLight(0xffffff, 0.8);
  mainLight.position.set(10, 20, -10);
  mainLight.castShadow = true;
  mainLight.shadow.mapSize.width = 1024;
  mainLight.shadow.mapSize.height = 1024;
  mainLight.shadow.camera.near = 1;
  mainLight.shadow.camera.far = 60;
  mainLight.shadow.camera.left = -20;
  mainLight.shadow.camera.right = 20;
  mainLight.shadow.camera.top = 20;
  mainLight.shadow.camera.bottom = -20;
  scene.add(mainLight);

  // Accent lights
  const accentLight1 = new THREE.PointLight(0x7cffcb, 1, 50);
  accentLight1.position.set(-10, 5, 0);
  scene.add(accentLight1);

  const accentLight2 = new THREE.PointLight(0xff6bcb, 1, 50);
  accentLight2.position.set(10, 5, 20);
  scene.add(accentLight2);

  // ─────────────────────────────────────────────────────────────
  // CREATE ARENA
  // ─────────────────────────────────────────────────────────────
  const court = SETTINGS.court;

  // Floor
  const floorGeo = new THREE.PlaneGeometry(court.width, court.depth);
  const floorMat = new THREE.MeshStandardMaterial({
    color: 0x0a0a1a,
    metalness: 0.8,
    roughness: 0.3,
  });
  const floor = new THREE.Mesh(floorGeo, floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.position.set(0, -court.height / 2, court.depth / 2);
  floor.receiveShadow = true;
  scene.add(floor);

  // Ceiling (subtle)
  const ceiling = new THREE.Mesh(floorGeo, new THREE.MeshStandardMaterial({
    color: 0x080818,
    metalness: 0.5,
    roughness: 0.8,
  }));
  ceiling.rotation.x = Math.PI / 2;
  ceiling.position.set(0, court.height / 2, court.depth / 2);
  scene.add(ceiling);

  // Side walls
  const wallGeo = new THREE.PlaneGeometry(court.depth, court.height);
  const wallMat = new THREE.MeshStandardMaterial({
    color: 0x0a0a2a,
    metalness: 0.6,
    roughness: 0.4,
    side: THREE.DoubleSide,
  });

  const leftWall = new THREE.Mesh(wallGeo, wallMat);
  leftWall.rotation.y = Math.PI / 2;
  leftWall.position.set(-court.width / 2, 0, court.depth / 2);
  scene.add(leftWall);

  const rightWall = new THREE.Mesh(wallGeo, wallMat);
  rightWall.rotation.y = -Math.PI / 2;
  rightWall.position.set(court.width / 2, 0, court.depth / 2);
  scene.add(rightWall);

  // Grid lines on floor
  const gridHelper = new THREE.GridHelper(Math.max(court.width, court.depth), 20, 0x1a1a3a, 0x1a1a3a);
  gridHelper.position.set(0, -court.height / 2 + 0.01, court.depth / 2);
  scene.add(gridHelper);

  // Center line (neon)
  const centerLineGeo = new THREE.BoxGeometry(court.width * 0.8, 0.05, 0.1);
  const centerLineMat = new THREE.MeshBasicMaterial({ color: 0x3a3a6a });
  const centerLine = new THREE.Mesh(centerLineGeo, centerLineMat);
  centerLine.position.set(0, -court.height / 2 + 0.02, court.depth / 2);
  scene.add(centerLine);

  // ─────────────────────────────────────────────────────────────
  // CREATE GAME OBJECTS
  // ─────────────────────────────────────────────────────────────
  const p = SETTINGS.paddle;

  // Player paddle (near camera)
  const paddleGeo = new THREE.BoxGeometry(p.width, p.height, p.depth);
  const playerMat = new THREE.MeshStandardMaterial({
    color: 0x00ffaa,
    emissive: 0x00ffaa,
    emissiveIntensity: 0.5,
    metalness: 0.8,
    roughness: 0.2,
  });
  const playerPaddle = new THREE.Mesh(paddleGeo, playerMat);
  playerPaddle.castShadow = true;
  scene.add(playerPaddle);

  // AI paddle (far end)
  const aiMat = new THREE.MeshStandardMaterial({
    color: 0xff66aa,
    emissive: 0xff66aa,
    emissiveIntensity: 0.5,
    metalness: 0.8,
    roughness: 0.2,
  });
  const aiPaddle = new THREE.Mesh(paddleGeo, aiMat);
  aiPaddle.castShadow = true;
  scene.add(aiPaddle);

  // Ball
  const ballGeo = new THREE.SphereGeometry(SETTINGS.ball.radius, 16, 16);
  const ballMat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    emissive: 0xffffaa,
    emissiveIntensity: 0.8,
    metalness: 0.9,
    roughness: 0.1,
  });
  const ballMesh = new THREE.Mesh(ballGeo, ballMat);
  ballMesh.castShadow = true;
  scene.add(ballMesh);

  // Ball glow
  const glowGeo = new THREE.SphereGeometry(SETTINGS.ball.radius * 2, 16, 16);
  const glowMat = new THREE.MeshBasicMaterial({
    color: 0xffffcc,
    transparent: true,
    opacity: 0.15,
  });
  const ballGlow = new THREE.Mesh(glowGeo, glowMat);
  scene.add(ballGlow);

  // Ball trail
  const trailPositions = [];
  const trailCount = 15;
  const trailMeshes = [];
  for (let i = 0; i < trailCount; i++) {
    const trailGeo = new THREE.SphereGeometry(SETTINGS.ball.radius * (1 - i / trailCount) * 0.7, 8, 8);
    const trailMat = new THREE.MeshBasicMaterial({
      color: 0xffffaa,
      transparent: true,
      opacity: 0.3 * (1 - i / trailCount),
    });
    const trailMesh = new THREE.Mesh(trailGeo, trailMat);
    trailMesh.visible = false;
    scene.add(trailMesh);
    trailMeshes.push(trailMesh);
    trailPositions.push(new THREE.Vector3());
  }

  // ─────────────────────────────────────────────────────────────
  // UTILITIES
  // ─────────────────────────────────────────────────────────────
  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  function randSign() {
    return Math.random() < 0.5 ? -1 : 1;
  }

  function formatOverlay(lines) {
    return lines.map((l) => `<div>${l}</div>`).join("");
  }

  function setOverlay(html, show = true) {
    overlayEl.innerHTML = html;
    overlayEl.classList.toggle("hidden", !show);
  }

  function setHint(text) {
    hintEl.textContent = text;
  }

  function updateScoreUI() {
    scoreLeftEl.textContent = String(State.scoreL);
    scoreRightEl.textContent = String(State.scoreR);
  }

  // ─────────────────────────────────────────────────────────────
  // GAME LOGIC
  // ─────────────────────────────────────────────────────────────
  function resetPositions() {
    const halfH = court.height / 2 - p.height / 2;

    // Player (near, z = 0)
    State.left.x = 0;
    State.left.y = 0;
    State.left.z = SETTINGS.paddle.inset;
    State.left.vx = 0;
    State.left.vy = 0;

    // AI (far, z = court.depth)
    State.right.x = 0;
    State.right.y = 0;
    State.right.z = court.depth - SETTINGS.paddle.inset;
    State.right.vx = 0;
    State.right.vy = 0;
    State.right.targetX = 0;
    State.right.targetY = 0;

    // Ball (center)
    State.ball.x = 0;
    State.ball.y = 0;
    State.ball.z = court.depth / 2;
    State.ball.vx = 0;
    State.ball.vy = 0;
    State.ball.vz = 0;
    State.ball.speed = 0;

    // Clear trail
    for (let i = 0; i < trailCount; i++) {
      trailMeshes[i].visible = false;
      trailPositions[i].set(State.ball.x, State.ball.y, State.ball.z);
    }
  }

  function resetMatch() {
    State.scoreL = 0;
    State.scoreR = 0;
    State.gameOver = false;
    State.paused = false;
    State.running = false;
    State.lastPointBy = 0;
    resetPositions();
    updateScoreUI();
    setHint("Space: serve • WASD or ↑/↓/←/→: move • P: pause • R: restart");
    setOverlay(
      formatOverlay([
        "PRESS SPACE TO SERVE",
        "<span style='opacity:.75;font-weight:600'>WASD or arrows to move • P pause • R restart</span>",
      ]),
      true
    );
  }

  function serve() {
    State.running = true;
    State.paused = false;
    setOverlay("", false);

    // Ball moves towards the AI (positive Z) or towards the player (negative Z)
    const dir = State.lastPointBy === 0 ? 1 : -State.lastPointBy;
    const angleY = (Math.random() * 0.5 - 0.25) * Math.PI; // Left/right variance
    const angleX = (Math.random() * 0.3 - 0.15) * Math.PI; // Up/down variance

    State.ball.speed = SETTINGS.ball.serveSpeed;
    State.ball.vz = Math.cos(angleY) * Math.cos(angleX) * State.ball.speed * dir;
    State.ball.vx = Math.sin(angleY) * State.ball.speed;
    State.ball.vy = Math.sin(angleX) * State.ball.speed * 0.3;
  }

  function finishPoint(winner) {
    State.lastPointBy = winner;
    State.running = false;

    if (winner === -1) State.scoreL += 1;
    else State.scoreR += 1;

    updateScoreUI();

    if (State.scoreL >= SETTINGS.winScore || State.scoreR >= SETTINGS.winScore) {
      State.gameOver = true;
      const won = State.scoreL > State.scoreR ? "YOU WIN!" : "AI WINS!";
      setOverlay(
        formatOverlay([
          won,
          "<span style='opacity:.75;font-weight:600'>Press R to restart</span>",
        ]),
        true
      );
      setHint("R: restart");
      beep("win");
      return;
    }

    resetPositions();
    setOverlay(
      formatOverlay([
        winner === -1 ? "YOU SCORED!" : "AI SCORED!",
        "<span style='opacity:.75;font-weight:600'>Press SPACE to serve</span>",
      ]),
      true
    );
    beep("score");
  }

  function togglePause() {
    if (State.gameOver) return;
    if (!State.running) return;
    State.paused = !State.paused;
    if (State.paused) {
      setOverlay(
        formatOverlay([
          "PAUSED",
          "<span style='opacity:.75;font-weight:600'>Press P to resume</span>",
        ]),
        true
      );
      setHint("P: resume • R: restart");
    } else {
      setOverlay("", false);
      setHint("Space: serve • WASD or ↑/↓/←/→: move • P: pause • R: restart");
    }
  }

  function collideWithPaddle(paddle, side) {
    const ball = State.ball;
    const b = SETTINGS.ball;
    const pd = SETTINGS.paddle;

    // Paddle bounds
    const px = paddle.x;
    const py = paddle.y;
    const pz = paddle.z;

    // Check if ball is within paddle's X-Y bounds and at paddle's Z
    const hitX = Math.abs(ball.x - px) < pd.width / 2 + b.radius;
    const hitY = Math.abs(ball.y - py) < pd.height / 2 + b.radius;

    let hitZ = false;
    if (side === -1) {
      // Player paddle (near end, z small)
      hitZ = ball.z - b.radius <= pz + pd.depth / 2 && ball.z > pz;
    } else {
      // AI paddle (far end, z large)
      hitZ = ball.z + b.radius >= pz - pd.depth / 2 && ball.z < pz;
    }

    if (!hitX || !hitY || !hitZ) return;

    // Only bounce if moving towards paddle
    if (side === -1 && ball.vz >= 0) return;
    if (side === +1 && ball.vz <= 0) return;

    // Calculate bounce angle based on where ball hit paddle
    const offsetY = clamp((ball.y - py) / (pd.height / 2), -1, 1);
    const offsetX = clamp((ball.x - px) / (pd.width / 2), -1, 1);

    const maxBounceAngle = 0.4 * Math.PI;

    // Speed up
    ball.speed = clamp(
      ball.speed + SETTINGS.ball.speedUpPerHit,
      SETTINGS.ball.serveSpeed,
      SETTINGS.ball.maxSpeed
    );

    // Reflect Z direction
    const dirZ = side === -1 ? 1 : -1;

    // New velocity
    ball.vz = Math.cos(offsetY * maxBounceAngle) * ball.speed * dirZ;
    ball.vy = Math.sin(offsetY * maxBounceAngle) * ball.speed * 0.5;
    ball.vx = offsetX * ball.speed * 0.3;

    // Nudge ball away from paddle
    if (side === -1) {
      ball.z = pz + pd.depth / 2 + b.radius + 0.1;
    } else {
      ball.z = pz - pd.depth / 2 - b.radius - 0.1;
    }

    beep("paddle");
  }

  function step(dt) {
    const halfW = court.width / 2;
    const halfH = court.height / 2 - SETTINGS.paddle.height / 2;
    const b = SETTINGS.ball;

    // Player input (vertical: W/S or ↑/↓)
    const up = Keys.up ? 1 : 0;
    const down = Keys.down ? 1 : 0;
    const intentY = down - up;
    State.left.vy = intentY * SETTINGS.paddle.speed;
    State.left.y = clamp(
      State.left.y + State.left.vy * dt,
      -halfH,
      halfH
    );

    // Player input (horizontal: A/D or ←/→)
    const left = Keys.left ? 1 : 0;
    const right = Keys.right ? 1 : 0;
    const intentX = left - right;  // Inverted to match camera perspective
    State.left.vx = intentX * SETTINGS.paddle.speed;
    const halfPaddleW = SETTINGS.paddle.width / 2;
    State.left.x = clamp(
      State.left.x + State.left.vx * dt,
      -halfW + halfPaddleW,
      halfW - halfPaddleW
    );

    // AI movement (tracks ball on both X and Y axes)
    const targetY = State.ball.y;
    const targetX = State.ball.x;
    State.right.targetY += (targetY - State.right.targetY) * SETTINGS.ai.reaction;
    State.right.targetX += (targetX - State.right.targetX) * SETTINGS.ai.reaction;

    const dead = SETTINGS.ai.deadZone;

    // AI Y movement
    const deltaY = State.right.targetY - State.right.y;
    const aiMoveY =
      Math.abs(deltaY) < dead
        ? 0
        : clamp(deltaY, -SETTINGS.ai.maxSpeed * dt, SETTINGS.ai.maxSpeed * dt);
    State.right.y = clamp(State.right.y + aiMoveY, -halfH, halfH);

    // AI X movement
    const deltaX = State.right.targetX - State.right.x;
    const aiMoveX =
      Math.abs(deltaX) < dead
        ? 0
        : clamp(deltaX, -SETTINGS.ai.maxSpeed * dt, SETTINGS.ai.maxSpeed * dt);
    const halfPaddleW2 = SETTINGS.paddle.width / 2;
    State.right.x = clamp(State.right.x + aiMoveX, -halfW + halfPaddleW2, halfW - halfPaddleW2);

    if (!State.running || State.paused || State.gameOver) return;

    // Update trail positions
    for (let i = trailCount - 1; i > 0; i--) {
      trailPositions[i].copy(trailPositions[i - 1]);
    }
    trailPositions[0].set(State.ball.x, State.ball.y, State.ball.z);

    // Ball motion
    State.ball.x += State.ball.vx * dt;
    State.ball.y += State.ball.vy * dt;
    State.ball.z += State.ball.vz * dt;

    // Wall bounces (X - left/right walls)
    const wallX = halfW - b.radius;
    if (State.ball.x <= -wallX) {
      State.ball.x = -wallX;
      State.ball.vx = Math.abs(State.ball.vx);
      beep("wall");
    } else if (State.ball.x >= wallX) {
      State.ball.x = wallX;
      State.ball.vx = -Math.abs(State.ball.vx);
      beep("wall");
    }

    // Ceiling/floor bounces (Y)
    const wallY = court.height / 2 - b.radius;
    if (State.ball.y <= -wallY) {
      State.ball.y = -wallY;
      State.ball.vy = Math.abs(State.ball.vy);
      beep("wall");
    } else if (State.ball.y >= wallY) {
      State.ball.y = wallY;
      State.ball.vy = -Math.abs(State.ball.vy);
      beep("wall");
    }

    // Paddle collisions
    collideWithPaddle(State.left, -1);
    collideWithPaddle(State.right, +1);

    // Scoring (Z bounds)
    if (State.ball.z < 0) {
      finishPoint(+1); // AI scores
    } else if (State.ball.z > court.depth) {
      finishPoint(-1); // Player scores
    }
  }

  function updateMeshPositions() {
    // Player paddle
    playerPaddle.position.set(State.left.x, State.left.y, State.left.z);

    // AI paddle
    aiPaddle.position.set(State.right.x, State.right.y, State.right.z);

    // Ball
    ballMesh.position.set(State.ball.x, State.ball.y, State.ball.z);
    ballGlow.position.copy(ballMesh.position);

    // Trail
    if (State.running && !State.paused) {
      for (let i = 0; i < trailCount; i++) {
        trailMeshes[i].visible = true;
        trailMeshes[i].position.copy(trailPositions[i]);
      }
    }

    // Pulse ball glow based on speed
    const speedRatio = State.ball.speed / SETTINGS.ball.maxSpeed;
    ballMat.emissiveIntensity = 0.5 + speedRatio * 0.5;
    glowMat.opacity = 0.1 + speedRatio * 0.1;
    ballGlow.scale.setScalar(1 + speedRatio * 0.5);
  }

  // ─────────────────────────────────────────────────────────────
  // AUDIO
  // ─────────────────────────────────────────────────────────────
  let audio;
  function ensureAudio() {
    if (audio) return audio;
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return null;
    audio = new Ctx();
    return audio;
  }

  function beep(kind) {
    const ac = ensureAudio();
    if (!ac) return;
    const now = ac.currentTime;
    const o = ac.createOscillator();
    const g = ac.createGain();

    const preset =
      kind === "paddle"
        ? { f: 520, d: 0.05 }
        : kind === "wall"
          ? { f: 380, d: 0.04 }
          : kind === "score"
            ? { f: 260, d: 0.09 }
            : kind === "win"
              ? { f: 740, d: 0.12 }
              : { f: 440, d: 0.04 };

    o.type = "square";
    o.frequency.setValueAtTime(preset.f, now);
    g.gain.setValueAtTime(0.0001, now);
    g.gain.exponentialRampToValueAtTime(0.06, now + 0.005);
    g.gain.exponentialRampToValueAtTime(0.0001, now + preset.d);

    o.connect(g).connect(ac.destination);
    o.start(now);
    o.stop(now + preset.d + 0.02);
  }

  // ─────────────────────────────────────────────────────────────
  // MAIN LOOP
  // ─────────────────────────────────────────────────────────────
  function loop(t) {
    if (!State.lastT) State.lastT = t;
    const dt = Math.min(0.05, (t - State.lastT) / 1000);
    State.lastT = t;

    State.acc += dt;
    const FIXED = 1 / 120;
    while (State.acc >= FIXED) {
      step(FIXED);
      State.acc -= FIXED;
    }

    updateMeshPositions();
    renderer.render(scene, camera);
    requestAnimationFrame(loop);
  }

  // ─────────────────────────────────────────────────────────────
  // INPUT HANDLERS
  // ─────────────────────────────────────────────────────────────
  function onKeyDown(e) {
    if (e.code === "ArrowUp" || e.code === "KeyW") Keys.up = true;
    if (e.code === "ArrowDown" || e.code === "KeyS") Keys.down = true;
    if (e.code === "ArrowLeft" || e.code === "KeyA") Keys.left = true;
    if (e.code === "ArrowRight" || e.code === "KeyD") Keys.right = true;

    if (e.code === "Space") {
      e.preventDefault();
      if (State.gameOver) return;
      if (!State.running) serve();
    }

    if (e.code === "KeyP") togglePause();
    if (e.code === "KeyR") resetMatch();

    const ac = ensureAudio();
    if (ac && ac.state === "suspended") ac.resume().catch(() => { });
  }

  function onKeyUp(e) {
    if (e.code === "ArrowUp" || e.code === "KeyW") Keys.up = false;
    if (e.code === "ArrowDown" || e.code === "KeyS") Keys.down = false;
    if (e.code === "ArrowLeft" || e.code === "KeyA") Keys.left = false;
    if (e.code === "ArrowRight" || e.code === "KeyD") Keys.right = false;
  }

  function onPointerDown() {
    const ac = ensureAudio();
    if (ac && ac.state === "suspended") ac.resume().catch(() => { });
    if (State.gameOver) return;
    if (!State.running) serve();
  }

  function onResize() {
    const rect = container.getBoundingClientRect();
    camera.aspect = rect.width / rect.height;
    camera.updateProjectionMatrix();
    renderer.setSize(rect.width, rect.height);
  }

  // ─────────────────────────────────────────────────────────────
  // INIT
  // ─────────────────────────────────────────────────────────────
  function init() {
    window.addEventListener("keydown", onKeyDown, { passive: false });
    window.addEventListener("keyup", onKeyUp);
    container.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("resize", onResize);

    onResize();
    resetMatch();
    requestAnimationFrame(loop);
  }

  init();
})();
