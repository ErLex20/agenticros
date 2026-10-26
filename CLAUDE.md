# AgenticROS (hackathon-nvidia fork)

AgenticROS is a ROS 2 integration for AI agent platforms: a **core** (transport, types, config) plus **adapters** per platform. This fork is used as a submodule of `hackathon-nvidia` (`tools/agenticros`), where its main consumer is the **Nebius adapter** driving a simulated Unitree Go2 in Gazebo.

The parent repo's `CLAUDE.md` covers the container, the simulation and the end-to-end workflow. This file covers the TypeScript monorepo.

## Deployment context (hackathon-nvidia)

- Everything runs inside the DUA dev container (`/home/neo/workspace`); the `ros2` CLI **is** available there and fine for debugging.
- Transport: **local DDS** (`transport.mode = "local"`, rclnodejs, domain 0), not Zenoh. Config comes from `AGENTICROS_CONFIG_PATH` = `<workspace>/config/agenticros.json`, not `~/.agenticros/config.json`.
- Robot namespace `""`. Topics:
  - camera `/dottorcane/slam/zed_x_driver/left/image_rect_color` (Image, 1920×1080, ~110° HFOV)
  - depth `/dottorcane/slam/zed_x_driver/depth_distances` (32FC1)
  - cmd_vel `/dottorcane/go2_control/cmd_vel`
- Safety limits: 0.25 m/s linear and 0.5 rad/s angular (from the parent config).
- ROS packages are built by the parent workspace through the symlink `src/agenticros -> tools/agenticros/ros2_ws/src`. Do not run `colcon build` inside `ros2_ws/`. Build from the workspace root with the container alias `cbuild` (`colcon build --event-handlers console_direct+ --symlink-install --continue-on-error`), e.g. `cbuild --packages-select agenticros_msgs`, from a shell where `ros2init` has sourced the DUA underlay (interactive shell or `bash -ic`).
- Commit and push changes here (remote `ErLex20/agenticros`, branch `main`) before bumping the submodule pointer in the parent repo.

## Nebius adapter (`packages/agenticros-nebius/src/`)

Launched by the parent's `scripts/run_nebius_agent.sh`, which loads `.env` and sets the simulation defaults. It spawns the Claude Code MCP server (`../agenticros-claude-code/dist/index.js`) as a stdio child and reaches the robot **only** through its tools (`ros2_camera_snapshot`, `ros2_depth_distance`, `ros2_depth_corridor`, `ros2_imu_yaw`, `ros2_move_for`, `ros2_estop`), so the AgenticROS safety clamps always apply.

| File | Purpose |
|------|---------|
| `index.ts` | Entry point: env parsing (`NEBIUS_*`), MCP client spawn, always `ros2_estop` on exit |
| `agent.ts` | `GoalAgent`: observe → decide → act loop with Nemotron tool calls, bounded by steps, time and a full search turn; without a map, after each full turn without the target it can ask the vision model where to explore (`suggestExploration`, opt-in with `NEBIUS_EXPLORE_CHOICE=vision`, box models only; default `open`: confirmed doorways, then the most open direction). A camera snapshot timeout is a missed observation (no motion), not the end of the task |
| `perception.ts` | Target localization by the vision model (default `NEBIUS_PERCEPTION=vlm`; `yolo` adds tiled YOLO for COCO targets). The vision model returns a bounding box (Kimi-K2.6, default) or a numbered column (MiniCPM-V), per its profile; crop checks of attributes and of vision-model sightings at confirmation and arrival (the model names the object before the verdict, `VERIFY_SCHEMA`); `chooseExploreView`: the views of a search turn in one mosaic, the model returns the view and the point to walk toward |
| `vision.ts` | Pure helpers: per-model vision profiles (grounding, image width, provider fields such as disabling Kimi's thinking), robust JSON extraction, box normalization, reply schema |
| `policy.ts` | Pure helpers (camera geometry, intent → bounded motion, goal test, deterministic fallback, search and exploration state). The no-map graph lives here: dead-reckoned position (`xM`, `yM`), places where the robot searched a full turn with their explored headings (`enterPlace`), and `suggestedHeadingDeg`, which `chooseExploreHeading` prefers. Also the `detour` around an obstacle between the robot and a confirmed target: when two motions in a row leave the target where it was in the image (stuck: turn away from the target's side) or when there is no room to advance toward it (blocked: the target's side first, then the other), turn 40° and walk up to 0.7 m; at most 3 per spot. With an IMU (`NEBIUS_IMU_TOPIC`) `applyImuYaw` replaces the commanded turns with the measured one in the heading and the searched angle, and a rotation that turns less than half the command (`blockedTurn`) starts the `UNSTICK_STEPS` (step forward, turn the other way, forward, back, other way; then finish as stuck), with the controller picking the search direction from then on. No ROS, no network |
| `robot.ts` | Thin wrappers around the MCP tools; `robotPose` (`ros2_robot_pose`), `navigateToPose` (`ros2_navigate_to_pose`), `savePlace`, configurable explore action (`NEBIUS_EXPLORE_ACTION`); turn/forward efficiency compensation for the legged base; `imuYaw` (gyroscope heading), `depthCorridor` (obstacles lower than the camera, combined with the band above the horizon in `agent.ts`, `NEBIUS_DEPTH_CORRIDOR=off` disables it), `backUp` |
| `__tests__/policy.test.ts`, `__tests__/vision.test.ts` | Offline unit tests for `policy.ts` and `vision.ts` |

With Nav2 and a pose estimate (`AGENTICROS_POSE_TOPIC`, set by the parent's launcher for `GO2_TARGET=real|sim`), `agent.ts` executes far approaches (`nav2ApproachGoal` in `policy.ts`), exploration walks and frontier trips as Nav2 goals, remembers the target position in the map frame (`rememberTarget`) and, with `NEBIUS_SESSION_PLACES=on`, saves named places of the session (`mode=remember`, targets reached). Switches: `NEBIUS_NAV2_APPROACH`, `NEBIUS_NAV2_WALK`, `NEBIUS_SESSION_PLACES`.

Rules:
- Keep geometry and safety decisions in deterministic code (`policy.ts`). The LLM picks semantic intents (`search` / `face_target` / `approach_target` / `finish`); the controller bounds, clamps or rejects them.
- The model never sees ground-truth pose. `AGENTICROS_DEBUG_POSE_TOPIC` is logging-only diagnostics.
- Logs go to stderr (`[Goal]`, `[Step N]`, `[Scene]`, `[Decision]`, `[Explore]`, `[Graph]`, `[Heading]`, `[Timing]`); stdout carries only the final answer in Italian.
- Never print or log `NEBIUS_API_KEY`.
- Reasoning and vision calls fall back to another NVIDIA model when the primary fails (`chat()` in `agent.ts`: Nemotron Nano with thinking off, 10 s decision budget; `Perception.ask()`: Nemotron-3.5-Super-VL, same grounding only); the fallback serves the next 5 calls.
- Every vision-model call goes through `Perception.ask()`, which applies the model profile from `vision.ts` (e.g. Kimi-K2.6 thinks by default and needs `chat_template_kwargs: {thinking: false}`). A new vision model needs a profile there and a measurement with the parent's `scripts/vision_eval/` before becoming the default.
- Geometry rules depend on `TargetObservation.localization` (`box` = tight box from YOLO or a grounding model, `column` = coarse band), not on `source`. `source` only drives trust (YOLO ≥ 0.8 confirmed at once, everything else needs two consecutive frames).
- Keep the arrival guards: `depthConsistentWithBox` (depth reading an occluder), the 0.5 m cap when the target distance is unknown, the crop check of vision-model sightings, and `closeFirstSighting` (a target first seen already within arrival distance, or at an unknown distance, filling the frame, is not confirmed by repetition: a sofa 0.37 m ahead passed for a bed). Each one fixed a false arrival seen in simulation (parent `docs/CHANGELOG.md`).
- The no-map exploration is measured offline with the parent's `scripts/vision_eval/explore_choice_eval.py` and in closed loop with `closed_loop_matrix.sh` (parent `docs/ESPLORAZIONE_GRAFO.md`); change its prompt or rules only with a new measurement.
- Commits: no Claude co-author (`Co-Authored-By`).

```bash
pnpm --filter @agenticros/nebius build      # required: the agent runs from dist/
pnpm --filter @agenticros/nebius test       # runs dist/__tests__, so build first
```

## Architecture

```
packages/
  core/                    # @agenticros/core — transport, types, Zod config (no platform deps)
  ros-camera/              # @agenticros/ros-camera — camera snapshot encoding (Image / CompressedImage)
  object-detection/        # @agenticros/object-detection — YOLOv8n COCO detector + find-object scan
  agenticros-nebius/       # @agenticros/nebius — Nebius Token Factory goal-loop agent (used here)
  agenticros-claude-code/  # @agenticros/claude-code — MCP server (stdio); also the Nebius agent's robot backend
  agenticros/              # @agenticros/agenticros — OpenClaw plugin (upstream, unused here)
  agenticros-gemini/       # @agenticros/gemini — Gemini CLI (upstream, unused here)
  agenticros-cli/          # agenticros — orchestrator CLI (`agenticros doctor`)
  robot-eyes/              # @agenticros/eyes — on-robot face display (upstream, unused here)
ros2_ws/src/
  agenticros_msgs/         # Custom ROS 2 messages & services
  agenticros_discovery/    # Capability discovery node (Python)
  agenticros_agent/        # WebRTC agent node (Python)
  agenticros_follow_me/    # Follow Me mission (Python)
```

### Core (`packages/core/src/`)
| File | Purpose |
|------|---------|
| `config.ts` | Zod config schema: transport modes, robot, safety, skills |
| `transport/factory.ts` | `createTransport(config)`: picks the implementation by mode |
| `transport/transport.ts` | `RosTransport` interface shared by all adapters |
| `transport/local/transport.ts` | Local DDS transport via rclnodejs (**used here**) |
| `transport/zenoh/`, `rosbridge/`, `webrtc/` | Other transports |
| `topic-utils.ts` | Namespace prefix helpers |

### Claude Code MCP server (`packages/agenticros-claude-code/src/`)
| File | Purpose |
|------|---------|
| `index.ts` | Entry point: `StdioServerTransport`, tool handlers |
| `tools.ts` | Tool definitions + handlers (`ros2_move_for`, `ros2_camera_snapshot`, `ros2_depth_distance`, `ros2_estop`, …) |
| `config.ts` | Config loading (env / `AGENTICROS_CONFIG_PATH` / `~/.agenticros/config.json`) |
| `safety.ts` | Velocity clamps applied before every publish |
| `depth.ts` | Depth image sampling; `getDepthCorridor`: nearest obstacle in a 3D corridor in front of the body (level camera, planar depth) |
| `tools.ts` (Nav2) | `ros2_robot_pose` (`readCurrentMapPose`: `AGENTICROS_POSE_TOPIC`, nav_msgs/Odometry by default, then `/amcl_pose`, `/pose`) and `ros2_navigate_to_pose` (`executeNavigateToPose` in core, the `navigate_to` capability; `ros2_estop` cancels it) |
| `imu-yaw.ts` | `ros2_imu_yaw`: heading change integrated from the IMU gyroscope (angular velocity projected on the vertical with roll and pitch; the absolute yaw is never used, in simulation it is the ground truth). Subscribes with a 500-message queue: with the default 10, messages were lost while the server converted a camera frame and the heading was off by up to 27° after 720° |

After editing it, rebuild with `pnpm --filter @agenticros/claude-code build`: both the MCP server and the Nebius agent run it from `dist/`.

## Conventions

- **ESM only**, TypeScript strict, NodeNext module resolution
- pnpm workspaces (`packages/*`), npm scope `@agenticros/`, ROS package prefix `agenticros_`
- Every transport implements `RosTransport` from `@agenticros/core`. The local transport honours `SubscribeOptions.queueLength` (QoS depth) above the default 10
- Config is validated with Zod and defaults live in the schema; never assume a field is set
- Shared ML or inference logic goes in a shared package (e.g. `object-detection`), not in per-adapter copies

## Build & development

```bash
pnpm install          # workspace deps (pinned pnpm 9.15.4)
pnpm build            # all packages
pnpm typecheck
pnpm test
pnpm --filter <pkg> build
```

The parent's `agenticros_setup` (`bin/setup_agenticros.sh`) runs the full setup: colcon build of the parent workspace plus pnpm install/build.

## Configuration

Load order: `AGENTICROS_CONFIG_PATH` → `~/.agenticros/config.json` → OpenClaw config. `AGENTICROS_ROBOT_NAMESPACE` overrides the namespace at runtime.

## Adding a ROS 2 tool

Add it to `packages/agenticros-claude-code/src/tools.ts` (definition + `callTool` handler); the Nebius agent then uses it through `robot.ts`. The upstream OpenClaw (`packages/agenticros/src/tools/`) and Gemini adapters mirror the same tool set; update them only if you intend to upstream the change.

## Safety

All velocity publishes go through the validator in `packages/agenticros-claude-code/src/safety.ts` (`maxLinearVelocity`, `maxAngularVelocity` from config). `ros2_move_for` always ends with a stop, and the MCP server stops in-flight motions on shutdown. Do not bypass these by publishing cmd_vel directly from adapter code.

## Docs

`docs/architecture.md`, `docs/cli.md`, `docs/cameras.md`, `docs/skills.md`, `docs/robot-setup.md`. Nebius and simulation docs live in the parent repo (`docs/NEBIUS_AGENTICROS.md`, `docs/GAZEBO_SETUP.md`, `src/README.md`).
