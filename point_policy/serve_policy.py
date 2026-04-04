"""
Server script for PointPolicy — runs in point-policy env.

Run with:
    python serve_policy.py \ 
    --bc_weight /home/haotian/Point-Policy/point_policy/exp_local/2026.04.03/point_policy/deterministic/143730_hidden_dim_256/snapshot/100000.pt \
    --port 8765 \
    --overrides "agent=point_policy" "suite=point_policy" "dataloader=point_policy" "suite.use_robot_points=true" "suite.use_object_points=true" "experiment=eval_point_policy" "suite/task/franka_env=pick_place_red_mug"
"""

import asyncio
import dataclasses
import json
import logging
import socket
import sys
from pathlib import Path
from typing import List

import hydra
import numpy as np
import torch
import tyro
import websockets
from omegaconf import OmegaConf

import utils
from eval_point_track import Workspace


@dataclasses.dataclass
class Args:
    bc_weight: str
    port: int = 8765
    config_path: str = "cfgs"
    config_name: str = "config"
    overrides: List[str] = dataclasses.field(default_factory=list)


def numpy_from_dict(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = numpy_from_dict(v)
        elif isinstance(v, list):
            out[k] = np.array(v)
        else:
            out[k] = v
    return out


def numpy_to_list(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy().tolist()
    if isinstance(v, dict):
        return {k: numpy_to_list(vv) for k, vv in v.items()}
    if isinstance(v, list):
        return [numpy_to_list(i) for i in v]
    return v


class PointPolicyServer:
    def __init__(self, args: Args):
        # mirror exactly what eval.py does
        with hydra.initialize(config_path=args.config_path, version_base=None):
            cfg = hydra.compose(
                config_name=args.config_name,
                overrides=args.overrides + [f"bc_weight={args.bc_weight}"],
            )

        logging.info("Config:\n%s", OmegaConf.to_yaml(cfg))

        self.workspace = Workspace(cfg)

        bc_snapshot = Path(args.bc_weight)
        if not bc_snapshot.exists():
            raise FileNotFoundError(f"bc weight not found: {bc_snapshot}")
        logging.info("Loading bc weight: %s", bc_snapshot)
        self.workspace.load_snapshot({"bc": bc_snapshot})
        self.workspace.agent.train(False)

        self.step = 0
        logging.info("PointPolicy loaded.")

    def reset(self):
        self.workspace.agent.buffer_reset()
        self.step = 0

    @torch.no_grad()
    def infer(self, obs: dict) -> np.ndarray:
        with utils.eval_mode(self.workspace.agent):
            action = self.workspace.agent.act(
                obs,
                self.workspace.expert_replay_loader.dataset.stats,
                self.step,
                self.step,
                eval_mode=True,
            )
        self.step += 1
        return action


def main(args: Args):
    server = PointPolicyServer(args)

    async def handle(websocket):
        logging.info("Client connected from %s", websocket.remote_address)
        async for message in websocket:
            data = json.loads(message)
            command = data.get("command", "infer")

            if command == "reset":
                server.reset()
                await websocket.send(json.dumps({"status": "reset"}))

            elif command == "infer":
                obs = numpy_from_dict(data["obs"])
                action = server.infer(obs)
                await websocket.send(json.dumps({
                    "action": numpy_to_list(action),
                }))

            else:
                await websocket.send(json.dumps({"error": f"Unknown command: {command}"}))

    async def serve():
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        logging.info("Serving on host=%s ip=%s port=%d", hostname, local_ip, args.port)
        async with websockets.serve(handle, "0.0.0.0", args.port):
            await asyncio.Future()

    asyncio.run(serve())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))