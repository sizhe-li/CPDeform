import numpy as np
from enum import Enum


class Reshape(Enum):
    COLLIDE_SINGLE_PRIMITIVE = 1
    CLAMP_DOUBLE_PRIMITIVE = 2


class Transport(Enum):
    PUSH_DOUBLE_PRIMITIVE = 1
    CLAMP_DOUBLE_PRIMITIVE = 2


ENV_TO_HEURISTIC = dict()

#############
## Reshape ##
#############

reshape_collide_envs = ['writer',
                        'torus',
                        'pinch',
                        'table',
                        'rollingpin',
                        'multistage_writer',
                        'print_letter',
                        ]

for env_name in reshape_collide_envs:
    ENV_TO_HEURISTIC[env_name] = Reshape.COLLIDE_SINGLE_PRIMITIVE

reshape_clamp_envs = ['multistage_airplane',
                      'multistage_chair',
                      'multistage_bottle',
                      'multistage_house',
                      'multistage_star',
                      'multistage_move',
                      ]

for env_name in reshape_clamp_envs:
    ENV_TO_HEURISTIC[env_name] = Reshape.CLAMP_DOUBLE_PRIMITIVE

###############
## Transport ##
###############

# transport_push_envs = ['rope']

# for env_name in transport_push_envs:
#     ENV_TO_HEURISTIC[env_name] = Transport.PUSH_DOUBLE_PRIMITIVE

transport_clamp_envs = ['rope',
                        'move',
                        'triplemove',
                        'assembly',
                        'multistage_rope',
                        ]

for env_name in transport_clamp_envs:
    ENV_TO_HEURISTIC[env_name] = Transport.CLAMP_DOUBLE_PRIMITIVE

MAX_ENV_STEPS = {
    'multistage_airplane': 1000000.,
    'multistage_chair': 1000000.,
    'multistage_bottle': 1000000.,
    'multistage_star':  1000000.,
    'multistage_move': 1000000.,
    'multistage_rope': 1000000.,
    'multistage_writer': 1000000.,
    'writer': 1000000.,
    'pinch': 1000000.,
}
