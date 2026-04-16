# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from lygra.robot.allegro import Allegro
from lygra.robot.shadow import Shadow
from lygra.robot.leap import Leap
from lygra.robot.dclaw import DClaw


class RobotFactory:
    _registry = {}

    @classmethod
    def register(cls, type_name, proc_cls):
        """Manual registration."""
        if type_name in cls._registry:
            raise KeyError(f"Robot type '{type_name}' is already registered with {cls._registry[type_name]}")
        cls._registry[type_name] = proc_cls

    @classmethod
    def register_decorator(cls, type_name):
        """Decorator-based registration."""
        def decorator(proc_cls):
            if type_name in cls._registry:
                raise KeyError(f"Robot type '{type_name}' is already registered with {cls._registry[type_name]}")
            cls._registry[type_name] = proc_cls
            return proc_cls
        return decorator

    @classmethod
    def create(cls, name, **kwargs):
        proc_cls = cls._registry.get(name)

        if proc_cls is None:
            raise ValueError(
                f"Robot type '{name}' is not supported. "
                f"Available types: {list(cls._registry.keys())}"
            )

        return proc_cls(**kwargs)


RobotFactory.register("allegro", Allegro)
RobotFactory.register("shadow", Shadow)
RobotFactory.register("leap", Leap)
RobotFactory.register("dclaw", DClaw)


def build_robot(name, urdf_path=None, **kwargs):
    return RobotFactory.create(name, urdf_path=urdf_path, **kwargs)
