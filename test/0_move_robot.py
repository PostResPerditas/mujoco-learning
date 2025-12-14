import mujoco
import mujoco.viewer
import time
import math

def main():
    # 1. 加载模型和初始化数据
    model = mujoco.MjModel.from_xml_path("./model/robot/aubo_i10/scene.xml")  # 请替换为您的XML文件路径
    data = mujoco.MjData(model)

    # 2. 设置您想要的关节值
    # 假设您的机器人有 n 个关节，您需要提供一个包含 n 个值的列表（单位：弧度）
    desired_joint_values = [0.0, -0.5, 0.0, 1.0, 0.0, 0.5]  # 示例值，请根据您的机器人修改

    # 确保设置的关节数量正确
    if len(desired_joint_values) == model.nq:
        data.qpos[:] = desired_joint_values  # 将期望的关节位置赋给data.qpos
    else:
        print(f"警告: 模型有 {model.nq} 个关节自由度，但提供了 {len(desired_joint_values)} 个值。")
        # 一种处理方式：只设置前 min(nq, len(values)) 个关节
        num_to_set = min(model.nq, len(desired_joint_values))
        data.qpos[:num_to_set] = desired_joint_values[:num_to_set]

    # 2.1 定义PD控制器参数
    kp = 100.0  # 比例增益（P）：决定关节回到目标位置的“刚度”或“力度”。值越大，抵抗偏差的力越大。
    kd = 10.0   # 微分增益（D）：决定系统的“阻尼”。值越大，抑制运动的速度越快，防止关节振荡。

    # 3. 使用前向动力学更新所有依赖位置的数据（如物体全局坐标）
    # 这一步很重要，它确保在仿真开始前，模型就位于您设置的初始姿态
    mujoco.mj_forward(model, data)

    # 4. 启动非阻塞查看器并运行仿真循环
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("仿真已启动。PD控制器正在工作以维持姿态。关闭查看器窗口以退出。")
        
        while viewer.is_running():
            # 对于每个关节，计算控制信号（扭矩）
            for i in range(len(desired_joint_values)):
                # 计算当前关节位置与目标位置的误差
                error_pos = desired_joint_values[i] - data.qpos[i]
                # 获取当前关节速度（我们希望速度误差为0，即静止）
                current_vel = data.qvel[i]
                
                # PD控制律：扭矩 = P * 位置误差 + D * （-当前速度）
                torque = kp * error_pos - kd * current_vel
                
                # 将计算出的扭矩赋给对应的执行器（控制指令）
                data.ctrl[i] = torque

            # 推进物理仿真一步（现在有了控制力矩，机器人会抵抗重力）
            mujoco.mj_step(model, data)

            # 同步查看器
            viewer.sync()
            time.sleep(0.01)

if __name__ == "__main__":
    main()