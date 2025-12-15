import mujoco
import mujoco.viewer
import time
import numpy as np
import os

class MultiSequenceController:
    def __init__(self, model, kp=100, kd=10, initial_positions=None):
        self.model = model
        self.data = mujoco.MjData(model)
        
        # 设置初始关节位置
        if initial_positions is not None:
            self.set_initial_positions(initial_positions)
        
        self.kp = kp
        self.kd = kd
        self.sequences = []
        self.current_sequence_index = 0
        self.current_step = 0
        self.is_running = False
        
    def set_initial_positions(self, positions):
        """设置机器手的初始关节位置"""
        if len(positions) <= self.model.nq:
            self.data.qpos[:len(positions)] = positions
            # 前向计算以更新模型状态
            mujoco.mj_forward(self.model, self.data)
            # print(f"初始位置已设置为: {positions}")
        else:
            print(f"警告: 提供的初始位置数量({len(positions)})超过模型关节数({self.model.nq})")
            self.data.qpos[:self.model.nq] = positions[:self.model.nq]
            mujoco.mj_forward(self.model, self.data)
    
    def add_sequence(self, target_positions, duration=1.0):  # 默认持续时间缩短为1秒
        """添加动作序列"""
        sequence = {
            'target_positions': np.array(target_positions),
            'duration': duration,
            'steps': int(duration / self.model.opt.timestep)
        }
        self.sequences.append(sequence)
        
    def pd_control(self, target_positions):
        """PD控制器"""
        for i in range(min(len(target_positions), self.model.nu)):
            error_pos = target_positions[i] - self.data.qpos[i]
            current_vel = self.data.qvel[i] if i < len(self.data.qvel) else 0.0
            torque = self.kp * error_pos - self.kd * current_vel
            if i < self.model.nu:
                self.data.ctrl[i] = torque
                
    def get_current_target(self):
        if self.current_sequence_index >= len(self.sequences):
            return None
        return self.sequences[self.current_sequence_index]['target_positions']
        
    def step(self):
        """执行单步仿真"""
        target = self.get_current_target()
        if target is None:
            return False
            
        self.pd_control(target)
        mujoco.mj_step(self.model, self.data)
        
        self.current_step += 1
        sequence = self.sequences[self.current_sequence_index]
        
        if self.current_step >= sequence['steps']:
            self.current_sequence_index += 1
            self.current_step = 0
            
        return self.current_sequence_index < len(self.sequences)

class CameraConfig:
    """相机配置类，用于存储和管理相机参数"""
    
    def __init__(self, distance=2.0, azimuth=45.0, elevation=-30.0, lookat=None):
        self.distance = distance      # 相机到目标的距离
        self.azimuth = azimuth        # 水平旋转角度（0-360度）
        self.elevation = elevation    # 垂直角度（-90到90度）
        self.lookat = lookat if lookat is not None else np.array([0.0, 0.0, 0.0])
    
    def apply_to_viewer(self, viewer):
        """将相机配置应用到查看器"""
        viewer.cam.distance = self.distance
        viewer.cam.azimuth = self.azimuth
        viewer.cam.elevation = self.elevation
        viewer.cam.lookat = self.lookat.copy()
        
    def __str__(self):
        return f"Camera: distance={self.distance}, azimuth={self.azimuth}, elevation={self.elevation}, lookat={self.lookat}"

def calculate_optimal_camera():
    """根据机器手位置计算最优相机参数"""
    
    return CameraConfig(
        distance=2,
        azimuth=225,
        elevation=-40,
        lookat=np.array([0, 1, 0.2])
    )

def move_joint_sequence(model_path, joint_sequence, simulation_speed=1.0):
    """改进的关节序列运动函数
    
    Args:
        model_path: 模型文件路径
        joint_sequence: 关节序列数组
        simulation_speed: 仿真速度倍数 (1.0为正常速度，>1.0为加速)
    """
    # 加载模型
    model = mujoco.MjModel.from_xml_path(model_path)
    
    # 使用第一个关节位置作为初始位置
    initial_positions = joint_sequence[0,:] if joint_sequence.shape[0] > 0 else None
    
    # 创建控制器并设置初始位置
    controller = MultiSequenceController(model, kp=25, kd=20, initial_positions=initial_positions)
    
    # 添加所有序列到控制器（缩短持续时间以加快运动）
    base_duration = 1.0  # 基础持续时间（秒）
    for i in range(joint_sequence.shape[0]):
        # 可以根据需要调整每个序列的持续时间
        duration = base_duration / simulation_speed  # 根据速度系数调整持续时间
        controller.add_sequence(joint_sequence[i,:], duration=duration)
    
    print(f"已添加 {len(controller.sequences)} 个动作序列")
    print(f"仿真速度: {simulation_speed}x")
    
    # 计算睡眠时间以控制仿真速度
    sleep_time = 0.001 / simulation_speed
    
    # 启动仿真
    with mujoco.viewer.launch_passive(controller.model, controller.data) as viewer:
        # 自动计算相机参数
        auto_camera = calculate_optimal_camera()
        auto_camera.apply_to_viewer(viewer)

        print("仿真开始! 机器手已初始化到第一个关节位置")
        try:
            while viewer.is_running():
                if not controller.step():
                    # 所有序列完成后的处理
                    pass
                    
                viewer.sync()
                time.sleep(max(sleep_time, 0.0001))  # 防止睡眠时间过短
                
        except KeyboardInterrupt:
            print("仿真被用户中断")

# 修改您的加载函数
def load_linker(speed_multiplier=2.0):
    """加载linker手模型并进行仿真
    
    Args:
        speed_multiplier: 仿真速度倍数，大于1加速，小于1减速
    """
    # model_path = 'model/robot/linker_hand/scene.xml'
    model_path = 'model/robot/linker_hand/object_on_table.xml'
    load_dir = 'assets/trajectory/'
    
    joint_sequence = np.loadtxt(os.path.join(load_dir, "test.txt"))
    print(f"成功加载轨迹文件，包含 {joint_sequence.shape[0]} 个位置点")
    
    # 调用改进的仿真函数
    move_joint_sequence(model_path, joint_sequence, simulation_speed=speed_multiplier)
        
def main():
    # 可以调整速度倍数来控制仿真速度
    # 1.0 = 正常速度, 2.0 = 2倍速度, 0.5 = 半速
    load_linker(speed_multiplier=2.0)  # 以2倍速度运行

if __name__ == "__main__":
    main()