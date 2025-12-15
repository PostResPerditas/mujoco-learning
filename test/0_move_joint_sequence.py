import mujoco
import mujoco.viewer
import time
import numpy as np
import os

class MultiSequenceController:
    def __init__(self, model, kp=100, kd=10):
        self.model = model
        self.data = mujoco.MjData(model)
        self.kp = kp  # 比例增益
        self.kd = kd  # 微分增益
        self.sequences = []  # 存储所有动作序列
        self.current_sequence_index = 0
        self.current_step = 0
        self.is_running = False
        
    def add_sequence(self, target_positions, duration=3.0):
        """添加一个动作序列"""
        sequence = {
            'target_positions': np.array(target_positions),
            'duration': duration,
            'steps': int(duration / self.model.opt.timestep)
        }
        self.sequences.append(sequence)
        
    def pd_control(self, target_positions):
        """PD控制器计算关节力矩[3](@ref)"""
        for i in range(len(target_positions)):
            error_pos = target_positions[i] - self.data.qpos[i]
            current_vel = self.data.qvel[i] if i < len(self.data.qvel) else 0.0
            torque = self.kp * error_pos - self.kd * current_vel
            if i < self.model.nu:
                self.data.ctrl[i] = torque
                
    def get_current_target(self):
        """获取当前目标位置"""
        if self.current_sequence_index >= len(self.sequences):
            return None
            
        sequence = self.sequences[self.current_sequence_index]
        return sequence['target_positions']
        
    def step(self):
        """执行单步仿真"""
        target = self.get_current_target()
        if target is None:
            return False
            
        # 应用PD控制
        self.pd_control(target)
        
        # 推进仿真
        mujoco.mj_step(self.model, self.data)
        
        self.current_step += 1
        sequence = self.sequences[self.current_sequence_index]
        
        # 检查当前序列是否完成
        if self.current_step >= sequence['steps']:
            self.current_sequence_index += 1
            self.current_step = 0
            # print(f"序列 {self.current_sequence_index} 完成")
            
        return self.current_sequence_index < len(self.sequences)

def move_joint_sequence(model_path, joint_sequence):

    # 加载模型
    model = mujoco.MjModel.from_xml_path(model_path)
    controller = MultiSequenceController(model, kp=25, kd=20)
    
    joint_sequence = joint_sequence
    
    # 添加所有序列到控制器
    for i in range(joint_sequence.shape[0]):
        controller.add_sequence(joint_sequence[i,:], duration=3.0)
    
    print(f"已添加 {len(controller.sequences)} 个动作序列")
    
    # 启动仿真
    with mujoco.viewer.launch_passive(controller.model, controller.data) as viewer:
        print("仿真开始!")
        try:
            while viewer.is_running():
                # 执行仿真步进，如果返回False说明所有序列已完成
                if not controller.step():
                    # print("所有动作序列执行完成!")
                    # 完成后保持窗口打开一段时间
                    # time.sleep(3)
                    # break
                    pass
                    
                viewer.sync()
                time.sleep(0.001)  # 控制仿真速度
                
        except KeyboardInterrupt:
            print("仿真被用户中断")

def load_aubo():
    model_path = 'model/robot/aubo_i10/scene.xml'
    joint_sequence = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.5, -0.3, 0.2, 0.1, -0.2, 0.1],
                               [0.8, -0.5, 0.4, 0.3, -0.4, 0.3],
                               [1.0, -0.7, 0.6, 0.5, -0.6, 0.5]])
    move_joint_sequence(model_path, joint_sequence)

def load_aubo_linker():
    model_path = 'model/robot/aubo_linker/scene.xml'
    joint_sequence = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0,]])
    move_joint_sequence(model_path, joint_sequence)

def load_linker():
    model_path = 'model/robot/linker_hand/scene.xml'

    load_dir = 'assets/trajectory/'
    joint_sequence = np.loadtxt(os.path.join(load_dir, "test.txt"))

    move_joint_sequence(model_path, joint_sequence)

def main():
    # load_aubo()
    # load_aubo_linker()
    load_linker()

if __name__ == "__main__":
    main()