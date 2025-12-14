import mujoco_viewer
import time
import mujoco
import numpy as np
import random

class Env(mujoco_viewer.CustomViewer):
    def __init__(self, path):
        super().__init__(path, 3, azimuth=-45, elevation=-30)
        self.path = path

if __name__ == "__main__":
    # env = Env("./model/franka_emika_panda/scene.xml")
    # env = Env("./model/robot/aubolink/scene.xml")
    # env = Env("./model/franka_emika_panda/panda.xml")
    env = Env("./model/franka_emika_panda/panda_usr.xml")
    # env = Env("./model/robot/aubo_i10/mujoco/scene.xml")
    # env = Env("./model/robot/aubo_i5/scene.xml")
        
    env.run_loop()