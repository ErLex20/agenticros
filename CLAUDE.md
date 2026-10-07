# AgenticROS (hackathon-nvidia fork)

AgenticROS is a ROS 2 integration for AI agent platforms: a **core**
(transport, types, config) plus **adapters** per platform. This fork is used as
a submodule of `hackathon-nvidia` (`tools/agenticros`), where its main consumer
is the **Nebius adapter** driving a simulated Unitree Go2 in Gazebo.

The parent repo's `CLAUDE.md` covers the container, the simulation and the
end-to-end workflow. This file covers the TypeScript monorepo.

## Deployment context (hackathon-nvidia)

- Everything runs inside the DUA dev container (`/home/neo/workspace`); the
  `ros2` CLI **is** available there and fine for debugging.
- Transport: **local DDS** (`transport.mode = "local"`, rclnodejs, domain 0),
  not Zenoh. Config comes from `AGENTICROS_CONFIG_PATH` =
  `<workspace>/config/agenticros.json`, not `~/.agenticros/config.json`.
- Robot namespace `""`. Topics:
  - camera `/dottorcane/slam/zed_x_driver/left/image_rect_color` (Image, 1920×1080, ~110° HFOV)
  - depth `/dottorcane/slam/zed_x_driver/depth_distances` (32FC1)
  - cmd_vel `/dottorcane/go2_control/cmd_vel`
- Safety limits: 0.25 m/s linear and 0.5 rad/s angular (from the parent config).
- ROS packages are built by the parent workspace through the symlink
  `src/agenticros -> tools/agenticros/ros2_ws/src`. Do not run `colcon build`
  inside `ros2_ws/`. Build from the workspace root with the container alias
  `cbuild` (`colcon build --symlink-install --continue-on-error`), e.g.
  `cbuild --packages-select agenticros_msgs`, from a shell where `ros2init`
  has sourced the DUA underlay (interactive shell or `bash -ic`).
- Commit and push changes here (remote `ErLex20/agenticros`, branch `main`)
  before bumping the submodule pointer in the parent repo.

## Nebius adapter (`packages/agenticros-nebius/src/`)

Launched by the parent's `scripts/run_nebius_agent.sh`, which loads `.env` and
sets the simulation defaults. It spawns the Claude Code MCP server
(`../agenticros-claude-code/dist/index.js`) as a stdio child and reaches the
robot **only** through its tools (`ros2_camera_snapshot`,
`ros2_depth_distance`, `ros2_move_for`, `ros2_estop`), so the AgenticROS safety
clamps always apply.

| File | Purpose |
|------|---------|
| `index.ts` | Entry point: env parsing (`NEBIUS_*`), MCP client spawn, always `ros2_estop` on exit |
| `agent.ts` | `GoalAgent`: observe → decide → act loop with Nemotron tool calls, bounded by steps, time and a full search turn |
| `perception.ts` | Tiled YOLO for COCO targets; MiniCPM-V with a numbered column grid for non-COCO targets and colour checks; depth sampling |
| `policy.ts` | Pure helpers (camera geometry, intent → bounded motion, goal test, deterministic fallback). No ROS, no network |
| `robot.ts` | Thin wrappers around the MCP tools; turn/forward efficiency compensation for the legged base |
| `__tests__/policy.test.ts` | Offline unit tests for `policy.ts` |

Rules:
- Keep geometry and safety decisions in deterministic code (`policy.ts`). The
  LLM picks semantic intents (`search` / `face_target` / `approach_target` /
  `finish`); the controller bounds, clamps or rejects them.
- The model never sees ground-truth pose. `AGENTICROS_DEBUG_POSE_TOPIC` is
  logging-only diagnostics.
- Logs go to stderr (`[Goal]`, `[Step N]`, `[Scene]`, `[Decision]`,
  `[Timing]`); stdout carries only the final answer in Italian.
- Never print or log `NEBIUS_API_KEY`.

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
| `depth.ts` | Depth image sampling |

After editing it, rebuild with `pnpm --filter @agenticros/claude-code build`:
both the MCP server and the Nebius agent run it from `dist/`.

## Conventions

- **ESM only**, TypeScript strict, NodeNext module resolution
- pnpm workspaces (`packages/*`), npm scope `@agenticros/`, ROS package prefix `agenticros_`
- Every transport implements `RosTransport` from `@agenticros/core`
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

The parent's `agenticros_setup` (`bin/setup_agenticros.sh`) runs the full
setup: colcon build of the parent workspace plus pnpm install/build.

## Configuration

Load order: `AGENTICROS_CONFIG_PATH` → `~/.agenticros/config.json` → OpenClaw
config. `AGENTICROS_ROBOT_NAMESPACE` overrides the namespace at runtime.

## Adding a ROS 2 tool

Add it to `packages/agenticros-claude-code/src/tools.ts` (definition +
`callTool` handler); the Nebius agent then uses it through `robot.ts`. The
upstream OpenClaw (`packages/agenticros/src/tools/`) and Gemini adapters mirror
the same tool set; update them only if you intend to upstream the change.

## Safety

All velocity publishes go through the validator in
`packages/agenticros-claude-code/src/safety.ts` (`maxLinearVelocity`,
`maxAngularVelocity` from config). `ros2_move_for` always ends with a stop, and
the MCP server stops in-flight motions on shutdown. Do not bypass these by
publishing cmd_vel directly from adapter code.

## Docs

`docs/architecture.md`, `docs/cli.md`, `docs/cameras.md`, `docs/skills.md`,
`docs/robot-setup.md`. Nebius and simulation docs live in the parent repo
(`docs/NEBIUS_AGENTICROS.md`, `docs/GAZEBO_SETUP.md`, `src/README.md`).
