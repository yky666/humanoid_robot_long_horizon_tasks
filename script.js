const robots = [
  {
    name: "松延 N2 / N2 EDU",
    tag: "目标平台",
    dof: "18 DoF",
    eef: "双臂+末端夹爪",
    safety: "关节限位+跌倒保护",
    asset: "assets/models/noetix_n2/urdf/N2.urdf",
    assetState: "松延/Noetix 官方 N2 URDF 已导入",
    route: "GVHMR/GMR → N2 官方 URDF RobotAdapter → LeRobot/GR00T",
    source: "songyan_n2_task_episode.npz",
    sketch: "n2",
    color: "#0b8a92",
    clips: {
      teleop: "simple_bend_handover_teleop.mp4",
      retarget: "simple_bend_pick_teleop.mp4",
      finetune: "simple_open_faucet_teleop.mp4",
      evaluate: "simple_locomotion_pick_between_tables.mp4",
      resample: "simple_locomotion_pick_between_tables.mp4"
    }
  },
  {
    name: "Unitree G1",
    tag: "验证平台",
    dof: "29 DoF",
    eef: "双臂+灵巧手",
    safety: "限位+碰撞",
    asset: "assets/models/unitree_g1/g1_29dof_with_hand.urdf",
    assetState: "远端真实 URDF 可接入",
    route: "GVHMR/GMR/OmniRetarget → G1 29DoF → GR00T",
    source: "g1_teleop_episode.npz",
    sketch: "humanoid",
    color: "#2f8f61",
    clips: {
      teleop: "pico_teleop_translate_rotate.mp4",
      retarget: "gmr_g1_walk_retarget.mp4",
      finetune: "omni_new_task2_g1_render.mp4",
      evaluate: "gmr_g1_walk_retarget.mp4",
      resample: "simple_locomotion_pick_between_tables.mp4"
    }
  },
  {
    name: "Unitree Go2 / B2",
    tag: "四足平台",
    dof: "12 DoF",
    eef: "四足运动底盘",
    safety: "足端接触+姿态约束",
    asset: "assets/models/unitree_quadruped/go2_or_b2.urdf",
    assetState: "SDK/URDF 待接入",
    route: "运动轨迹 → Go2/B2 足端约束 → 四足数据集",
    source: "go2_b2_locomotion_episode.npz",
    sketch: "dog",
    color: "#bd7a22",
    clips: {
      teleop: "keyboard_teleop.mp4",
      retarget: "isaaclab_g1_velocity_rough.mp4",
      finetune: "ep0030_quality_replay.mp4",
      evaluate: "soccer_mosc_step3000.mp4",
      resample: "soccer_mosc_step3000.mp4"
    }
  },
  {
    name: "因时灵巧手 RH56DFTP",
    tag: "灵巧手",
    dof: "16-20 DoF",
    eef: "五指骨骼链",
    safety: "指关节限位+接触力边界",
    asset: "assets/models/inspire_hand/rh56dftp_hand.urdf",
    assetState: "因时灵巧手 RH56DFTP 数据与视频已接入",
    route: "MANO/手套骨骼 → 因时 RH56DFTP 指关节 → BC/RL 微调",
    source: "mano_to_rh56dftp_hand_pose.pkl",
    sketch: "hand",
    color: "#7952b3",
    clips: {
      teleop: "dexterous_hand_grasp.mp4",
      retarget: "inspire_hand_retarget.mp4",
      finetune: "rh56dftp_bc_rnn_eval.mp4",
      evaluate: "rh56dftp_ppo_eval.mp4",
      resample: "dexterous_hand_grasp.mp4"
    }
  }
];

const stages = [
  {
    key: "teleop",
    name: "遥操数采",
    short: "多源遥操输入到 Episode",
    title: "Teleop Data Capture",
    clip: {
      title: "PICO 遥操作数采",
      privacy: "发布前需复核",
      desc: "采集员通过 PICO/键盘/手套等输入设备产生机器人轨迹，系统同步记录 RGB-D、IMU、关节、控制指令、事件日志和自然语言任务。"
    },
    desc: "系统自动生成 TaskSpec，采集 RGB-D、关节、IMU、控制指令、语言任务和事件日志，并在采集中实时检查空帧、遮挡、同步漂移、动作抖动与复位次数。",
    artifacts: ["TaskSpec.json", "EpisodeSchema.parquet", "camera/imu/joint/control 同步流", "在线质量分 Q0"],
    metrics: ["18.4", "82", "76", "中"]
  },
  {
    key: "retarget",
    name: "重定向",
    short: "跨构型动作映射",
    title: "Cross-Embodiment Retargeting",
    clip: {
      title: "GMR 步态重定向",
      privacy: "可公开展示",
      desc: "RobotAdapter 根据目标机器人自由度、关节限位、末端约束和坐标系，把源轨迹自动映射到不同构型。"
    },
    desc: "通过 RobotAdapter 抽象关节、末端、传感器和安全约束，自动完成 G1、N2/N2 EDU、灵巧手或代理机器人之间的动作空间映射，并保留可追溯的转换日志。",
    artifacts: ["robot_adapter.yaml", "retarget_map.npz", "joint_limit_report.html", "失败帧自动标记"],
    metrics: ["22.1", "86", "79", "中"]
  },
  {
    key: "finetune",
    name: "后训练微调",
    short: "LeRobot / GR00T 数据闭环",
    title: "Post-training Fine-tuning",
    clip: {
      title: "GR00T Episode 0038",
      privacy: "可公开展示",
      desc: "合格轨迹被转换为 LeRobot/GR00T 兼容格式，进入 SFT、RL 或策略微调，并记录训练版本。"
    },
    desc: "系统筛选 Q 与 V 达标的轨迹，自动转换数据格式，生成训练集版本、训练配置和评估任务。低质量但高价值的失败片段不会丢弃，而是进入恢复性训练或对比样本池。",
    artifacts: ["LeRobotDataset v0.3", "GR00T finetune config", "failure-as-recovery samples", "model_card + run_id"],
    metrics: ["24.8", "88", "84", "高"]
  },
  {
    key: "evaluate",
    name: "评估诊断",
    short: "失败归因与价值评分",
    title: "Evaluation and Failure Diagnosis",
    clip: {
      title: "阶段评估回放",
      privacy: "可公开展示",
      desc: "评估结果回放 GT vs Pred，自动定位异常片段、阶段失败原因和模型不确定区域。"
    },
    desc: "模型评估结果被拆到阶段级：成功率、偏差、抖动、物体滑落、抓取失败、同步异常都会被归因到具体任务阶段，并计算质量分 Q 与价值分 V。",
    artifacts: ["stage_success.csv", "failure_heatmap.json", "Q/V score report", "uncertainty segments"],
    metrics: ["21.3", "91", "89", "高"]
  },
  {
    key: "resample",
    name: "Resample 回采",
    short: "主动回采进入下一轮",
    title: "Active Resampling",
    clip: {
      title: "跨桌移动抓取",
      privacy: "可公开展示",
      desc: "系统把失败热力图和场景覆盖缺口转成下一轮回采队列，优先补齐模型最不确定、最容易失败的长尾样本。"
    },
    desc: "Resample Agent 根据模型不确定性、失败频率、场景新颖性和业务优先级生成下一轮采集清单，自动调度机器人、任务变量、采集预算和验收规则。",
    artifacts: ["resample_queue.csv", "next_task_specs/", "scene_variable_plan", "采集预算与验收阈值"],
    metrics: ["31.6", "93", "92", "最高"]
  }
];

let currentRobot = 1;
let currentStage = 0;
let timer = null;

const robotGrid = document.querySelector("#robotGrid");
const pipeline = document.querySelector("#pipeline");
const video = document.querySelector("#demoVideo");
const stageName = document.querySelector("#stageName");
const stageTitle = document.querySelector("#stageTitle");
const stageDesc = document.querySelector("#stageDesc");
const clipTitle = document.querySelector("#clipTitle");
const clipDesc = document.querySelector("#clipDesc");
const privacyBadge = document.querySelector("#privacyBadge");
const artifactList = document.querySelector("#artifactList");
const dofValue = document.querySelector("#dofValue");
const eefValue = document.querySelector("#eefValue");
const safetyValue = document.querySelector("#safetyValue");
const runState = document.querySelector("#runState");
const mThroughput = document.querySelector("#mThroughput");
const mQuality = document.querySelector("#mQuality");
const mValue = document.querySelector("#mValue");
const mResample = document.querySelector("#mResample");
const sourceAsset = document.querySelector("#sourceAsset");
const targetRobot = document.querySelector("#targetRobot");
const robotSketch = document.querySelector("#robotSketch");
const smoothBar = document.querySelector("#smoothBar");
const assetStatus = document.querySelector("#assetStatus");
const assetPath = document.querySelector("#assetPath");
const retargetRoute = document.querySelector("#retargetRoute");
const viewUrdf = document.querySelector("#viewUrdf");
const urdfPreview = document.querySelector("#urdfPreview");

function renderRobots() {
  robotGrid.innerHTML = "";
  robots.forEach((robot, index) => {
    const button = document.createElement("button");
    button.className = "robot-button";
    button.innerHTML = `<span>${robot.name}</span><small>${robot.tag}</small>`;
    button.addEventListener("click", () => {
      currentRobot = index;
      updateRobot();
      setStage(currentStage);
    });
    robotGrid.appendChild(button);
  });
}

function renderPipeline() {
  pipeline.innerHTML = "";
  stages.forEach((stage, index) => {
    const button = document.createElement("button");
    button.className = "stage-button";
    button.innerHTML = `<strong>${index + 1}. ${stage.name}</strong><span>${stage.short}</span>`;
    button.addEventListener("click", () => {
      stopAutoRun();
      setStage(index);
    });
    pipeline.appendChild(button);
  });
}

function updateRobot() {
  const robot = robots[currentRobot];
  dofValue.textContent = robot.dof;
  eefValue.textContent = robot.eef;
  safetyValue.textContent = robot.safety;
  assetPath.textContent = robot.asset;
  retargetRoute.textContent = robot.route;
  assetStatus.classList.toggle("warn", robot.assetState.includes("待"));
  assetStatus.innerHTML = `<b>${robot.assetState}</b><br>${robot.asset}`;
  urdfPreview.classList.remove("visible");
  urdfPreview.textContent = "";
  targetRobot.textContent = `target: ${robot.name}`;
  document.querySelectorAll(".robot-button").forEach((button, index) => {
    button.classList.toggle("active", index === currentRobot);
  });
  drawRobotSketch(robot);
}

function setStage(index) {
  currentStage = index;
  const stage = stages[index];
  const robot = robots[currentRobot];
  const clipFile = robot.clips[stage.key];
  stageName.textContent = stage.name;
  stageTitle.textContent = stage.title;
  stageDesc.textContent = stage.desc;
  clipTitle.textContent = `${robot.name} · ${stage.clip.title}`;
  const assetNote =
    robot.name.includes("松延") ? "当前接入 N2 官方公开参数派生 URDF 预览，可点击“查看当前 URDF”检查 joints/limits；若获得 SDK 交付包，可直接替换为官方 n2.urdf。" : `当前目标构型由 ${robot.asset} 提供 DoF、关节限位、末端和安全约束。`;
  clipDesc.textContent = `${stage.clip.desc} 当前目标构型为 ${robot.name}，${assetNote}`;
  privacyBadge.textContent = stage.clip.privacy;
  sourceAsset.textContent = `source: ${robot.source}`;
  smoothBar.style.setProperty("--w", `${42 + index * 11}%`);
  video.src = `./assets/videos/${clipFile}`;
  artifactList.innerHTML = stage.artifacts
    .map((item) => `<div class="artifact"><b>✓</b><span>${item}</span></div>`)
    .join("");
  [mThroughput.textContent, mQuality.textContent, mValue.textContent, mResample.textContent] = stage.metrics;
  document.querySelectorAll(".stage-button").forEach((button, buttonIndex) => {
    button.classList.toggle("active", buttonIndex === index);
  });
  document.querySelectorAll(".flywheel-node").forEach((node) => {
    node.classList.toggle("active", node.dataset.node === stage.key);
  });
  video.play().catch(() => {});
}

function drawRobotSketch(robot) {
  const c = robot.color;
  const common = `<path d="M44 246 C118 190 194 232 262 170 S352 126 390 82" fill="none" stroke="${c}" stroke-width="4" stroke-dasharray="8 8" opacity=".85"/>`;
  const humanoid = `
    ${common}
    <circle cx="210" cy="54" r="19" fill="none" stroke="${c}" stroke-width="5"/>
    <path d="M210 76 L210 144 M158 104 L210 92 L264 106 M210 144 L174 220 M210 144 L248 220" fill="none" stroke="#dce8f4" stroke-width="8" stroke-linecap="round"/>
    <circle cx="158" cy="104" r="8" fill="${c}"/><circle cx="264" cy="106" r="8" fill="${c}"/>
    <circle cx="174" cy="220" r="8" fill="${c}"/><circle cx="248" cy="220" r="8" fill="${c}"/>
  `;
  const quadruped = `
    ${common}
    <path d="M126 132 L258 132 L300 166 M150 132 L116 198 M184 132 L172 204 M232 132 L242 204 M270 138 L306 204" fill="none" stroke="#dce8f4" stroke-width="8" stroke-linecap="round"/>
    <circle cx="306" cy="118" r="17" fill="none" stroke="${c}" stroke-width="5"/>
    <circle cx="116" cy="198" r="7" fill="${c}"/><circle cx="172" cy="204" r="7" fill="${c}"/><circle cx="242" cy="204" r="7" fill="${c}"/><circle cx="306" cy="204" r="7" fill="${c}"/>
  `;
  const dog = `
    ${common}
    <path d="M112 130 L242 124 L306 146 M132 130 L104 190 L92 222 M170 128 L158 190 L146 224 M226 126 L240 190 L232 224 M282 140 L318 188 L330 220" fill="none" stroke="#dce8f4" stroke-width="8" stroke-linecap="round"/>
    <path d="M286 126 L324 108 M120 126 L88 112" fill="none" stroke="#dce8f4" stroke-width="7" stroke-linecap="round"/>
    <circle cx="334" cy="104" r="18" fill="none" stroke="${c}" stroke-width="5"/>
    <circle cx="92" cy="222" r="7" fill="${c}"/><circle cx="146" cy="224" r="7" fill="${c}"/><circle cx="232" cy="224" r="7" fill="${c}"/><circle cx="330" cy="220" r="7" fill="${c}"/>
  `;
  const n2 = `
    ${common}
    <circle cx="210" cy="50" r="17" fill="none" stroke="${c}" stroke-width="5"/>
    <path d="M210 70 L210 148 M160 100 L210 88 L260 100 M160 100 L140 146 M260 100 L282 146 M210 148 L178 224 M210 148 L242 224" fill="none" stroke="#dce8f4" stroke-width="8" stroke-linecap="round"/>
    <path d="M188 118 L232 118 M190 136 L230 136" stroke="${c}" stroke-width="5" stroke-linecap="round"/>
    <circle cx="140" cy="146" r="8" fill="${c}"/><circle cx="282" cy="146" r="8" fill="${c}"/>
    <circle cx="178" cy="224" r="8" fill="${c}"/><circle cx="242" cy="224" r="8" fill="${c}"/>
  `;
  const hand = `
    ${common}
    <path d="M204 236 C180 200 176 156 190 118 L224 118 C238 156 232 200 214 236 Z" fill="none" stroke="#dce8f4" stroke-width="7"/>
    <path d="M190 120 L152 64 L144 38 M200 116 L190 64 L188 32 M212 116 L220 62 L226 30 M224 124 L260 74 L278 52 M188 166 L140 142 L112 136" fill="none" stroke="#dce8f4" stroke-width="7" stroke-linecap="round"/>
    <circle cx="152" cy="64" r="6" fill="${c}"/><circle cx="144" cy="38" r="7" fill="${c}"/>
    <circle cx="190" cy="64" r="6" fill="${c}"/><circle cx="188" cy="32" r="7" fill="${c}"/>
    <circle cx="220" cy="62" r="6" fill="${c}"/><circle cx="226" cy="30" r="7" fill="${c}"/>
    <circle cx="260" cy="74" r="6" fill="${c}"/><circle cx="278" cy="52" r="7" fill="${c}"/>
    <circle cx="140" cy="142" r="6" fill="${c}"/><circle cx="112" cy="136" r="7" fill="${c}"/>
  `;
  robotSketch.innerHTML =
    robot.sketch === "dog" ? dog : robot.sketch === "quadruped" ? quadruped : robot.sketch === "hand" ? hand : robot.sketch === "n2" ? n2 : humanoid;
}

function stepForward() {
  setStage((currentStage + 1) % stages.length);
}

function stopAutoRun() {
  if (timer) {
    clearInterval(timer);
    timer = null;
  }
  runState.textContent = "手动控制";
}

document.querySelector("#stepRun").addEventListener("click", () => {
  stopAutoRun();
  stepForward();
});

document.querySelector("#autoRun").addEventListener("click", () => {
  if (timer) {
    stopAutoRun();
    return;
  }
  runState.textContent = "自动闭环中";
  stepForward();
  timer = setInterval(stepForward, 6500);
});

viewUrdf.addEventListener("click", async () => {
  const robot = robots[currentRobot];
  try {
    const response = await fetch(robot.asset);
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    const text = await response.text();
    const jointCount = (text.match(/<joint\b/g) || []).length;
    const linkCount = (text.match(/<link\b/g) || []).length;
    const limitCount = (text.match(/<limit\b/g) || []).length;
    const head = text.split("\n").slice(0, 80).join("\n");
    urdfPreview.textContent = `${robot.name}\nasset: ${robot.asset}\nlinks: ${linkCount}, joints: ${jointCount}, limits: ${limitCount}\n\n${head}`;
  } catch (error) {
    urdfPreview.textContent = `${robot.name}\nasset: ${robot.asset}\nURDF 暂不可读取：${error.message}`;
  }
  urdfPreview.classList.add("visible");
});

renderRobots();
renderPipeline();
updateRobot();
setStage(0);
