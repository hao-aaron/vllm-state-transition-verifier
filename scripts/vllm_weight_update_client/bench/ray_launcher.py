#!/usr/bin/env python3
"""Run ``trainer_main.py``'s ranks as Ray actors instead of torchrun processes.

Takes the same arguments as ``trainer_main.py`` plus the Ray options below. One
actor per trainer rank is placed in a placement group (one GPU per bundle), so
trainer ranks can span nodes of the Ray cluster; the actors then form the
torch.distributed group themselves. Any engine runs this way, and engines whose
trainer ranks must be Ray actors (``sharded_rdt``) only run this way.

    ray_launcher.py --ray-address auto --ray-strategy SPREAD -- \
        --base-url http://host:8000 --backend nccl --layout fsdp=4 \
        --checkpoint <B> --checkpoint <C> --output result.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

HERE = Path(__file__).resolve().parent


@ray.remote(num_gpus=1)
class TrainerRank:
    def __init__(self) -> None:
        sys.path[:0] = [str(HERE), str(HERE.parent)]

    def endpoint(self) -> tuple[str, int]:
        from vllm.utils.network_utils import get_ip, get_open_port

        return get_ip(), get_open_port()

    def run(
        self, argv: list[str], rank: int, world_size: int, master: tuple[str, int]
    ) -> dict[str, Any] | None:
        import trainer_main

        os.environ.update(
            RANK=str(rank),
            WORLD_SIZE=str(world_size),
            LOCAL_RANK="0",  # Ray exposes exactly this actor's GPU
            MASTER_ADDR=master[0],
            MASTER_PORT=str(master[1]),
        )
        args = trainer_main.build_parser().parse_args(argv)
        return trainer_main.run(args, launcher="ray")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        usage="%(prog)s [ray options] -- [trainer_main.py args]",
    )
    parser.add_argument(
        "--ray-address", help="existing cluster (e.g. auto); default: start one"
    )
    parser.add_argument("--ray-namespace", default="wu-bench")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="environment variable for every trainer actor; repeatable",
    )
    parser.add_argument(
        "--ray-strategy",
        default="PACK",
        choices=("PACK", "SPREAD", "STRICT_PACK", "STRICT_SPREAD"),
    )
    argv = sys.argv[1:]
    if "--" not in argv:
        parser.error("pass trainer_main.py arguments after --")
    split = argv.index("--")
    args = parser.parse_args(argv[:split])
    trainer_argv = argv[split + 1 :]

    sys.path[:0] = [str(HERE), str(HERE.parent)]
    import trainer_main
    from layout import TrainerLayout

    trainer_args = trainer_main.build_parser().parse_args(trainer_argv)
    world_size = (
        1
        if trainer_args.layout == "trainerless"
        else TrainerLayout.parse(trainer_args.layout).world_size
    )

    ray.init(
        address=args.ray_address,
        namespace=args.ray_namespace,
        runtime_env={
            "env_vars": {
                "PYTHONPATH": f"{HERE}:{HERE.parent}",
                **dict(item.split("=", 1) for item in args.env),
            }
        },
    )
    group = placement_group(
        [{"GPU": 1, "CPU": 1}] * world_size, strategy=args.ray_strategy
    )
    ray.get(group.ready())
    actors = [
        TrainerRank.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=group, placement_group_bundle_index=rank
            )
        ).remote()
        for rank in range(world_size)
    ]
    master = ray.get(actors[0].endpoint.remote())
    results = ray.get(
        [
            actor.run.remote(trainer_argv, rank, world_size, master)
            for rank, actor in enumerate(actors)
        ]
    )
    trainer_main.write_result(results[0], trainer_args.output)


if __name__ == "__main__":
    main()
