import mujoco.viewer
import time
 
def main():
    # model = mujoco.MjModel.from_xml_path('model/robot/ur5e/scene.xml')
    model = mujoco.MjModel.from_xml_path('model/robot/aubo_i10/scene.xml')
    data = mujoco.MjData(model)
    data.ctrl[:6] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
 
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.002) # 让动画速度变慢，不然更新太快看不清机械臂的运动过程
 
if __name__ == "__main__":
    main()