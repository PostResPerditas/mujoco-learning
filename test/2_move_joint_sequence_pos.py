import mujoco
import mujoco.viewer
import time
import numpy as np
import os

class SimpleObjectImporter:
    """简化的物体导入器，专注于可视化"""
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.available_objects = ['potted_meat_can']
        
    def place_object(self, obj_name, position, quaternion=None):
        """在指定位置放置物体"""
        if obj_name not in self.available_objects:
            print(f"错误: 物体 '{obj_name}' 不可用。可用物体: {self.available_objects}")
            return False
            
        # 获取物体body ID
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        if body_id == -1:
            print(f"错误: 未找到物体 '{obj_name}'")
            return False
            
        # 隐藏所有其他物体
        self.hide_all_objects()
        
        # 设置目标物体位置
        start_idx = self.model.jnt_qposadr[self.model.body_jntadr[body_id]]
        
        # 设置位置 (x, y, z)
        self.data.qpos[start_idx:start_idx+3] = position
        
        # 设置姿态 (四元数)
        if quaternion is not None:
            self.data.qpos[start_idx+3:start_idx+7] = quaternion
        else:
            self.data.qpos[start_idx+3:start_idx+7] = np.array([1, 0, 0, 0])
            
        # 重置速度为零
        vel_start_idx = self.model.jnt_dofadr[self.model.body_jntadr[body_id]]
        self.data.qvel[vel_start_idx:vel_start_idx+6] = 0
        
        print(f"已放置物体 '{obj_name}' 到位置 {position}")
        return True
    
    def hide_all_objects(self):
        """隐藏所有物体（移到远处）"""
        hide_pos = np.array([10.0, 10.0, 10.0])
        for obj_name in self.available_objects:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
            if body_id != -1:
                start_idx = self.model.jnt_qposadr[self.model.body_jntadr[body_id]]
                self.data.qpos[start_idx:start_idx+3] = hide_pos

class MultiSequenceController:
    def __init__(self, model, kp=100, kd=10, initial_positions=None):
        self.model = model
        self.data = mujoco.MjData(model)
        
        # 设置初始关节位置
        if initial_positions is not None:
            self.set_initial_positions(initial_positions)
        
        # 调整PD控制参数，使其更适合位置控制
        # 降低刚度（kp），提高阻尼（kd）可以使运动更柔和
        self.kp = kp    # 位置增益（刚度）
        self.kd = kd    # 速度增益（阻尼）
        self.sequences = []
        self.current_sequence_index = 0
        self.current_step = 0
        self.is_running = False
        
        # 初始化物体导入器
        self.object_importer = SimpleObjectImporter(model, self.data)
        
    def set_initial_positions(self, positions):
        """设置机器手的初始关节位置"""
        if len(positions) <= self.model.nq:
            self.data.qpos[:len(positions)] = positions
            mujoco.mj_forward(self.model, self.data)
        else:
            print(f"警告: 提供的初始位置数量({len(positions)})超过模型关节数({self.model.nq})")
            self.data.qpos[:self.model.nq] = positions[:self.model.nq]
            mujoco.mj_forward(self.model, self.data)
    
    def add_sequence(self, target_positions, duration=1.0):
        """添加动作序列"""
        sequence = {
            'target_positions': np.array(target_positions),
            'duration': duration,
            'steps': int(duration / self.model.opt.timestep)
        }
        self.sequences.append(sequence)
        
    def position_control(self, target_positions):
        """纯位置控制器（关键修改点）"""
        for i in range(min(len(target_positions), self.model.nu)):
            # 核心修改：直接设置目标位置，而非计算力矩
            # MuJoCo的position执行器会根据此处设置的ctrl值作为目标位置
            # 并结合XML中定义的kp、kd参数自动进行PD控制
            self.data.ctrl[i] = target_positions[i]
                
    def get_current_target(self):
        if self.current_sequence_index >= len(self.sequences):
            return None
        return self.sequences[self.current_sequence_index]['target_positions']
        
    def step(self):
        """执行单步仿真（修改控制方法调用）"""
        target = self.get_current_target()
        if target is None:
            return False
            
        # 关键修改：使用位置控制而非PD控制
        self.position_control(target)
        mujoco.mj_step(self.model, self.data)
        
        self.current_step += 1
        if self.current_sequence_index < len(self.sequences):
            sequence = self.sequences[self.current_sequence_index]
            
            if self.current_step >= sequence['steps']:
                self.current_sequence_index += 1
                self.current_step = 0
            
        return self.current_sequence_index < len(self.sequences)

def move_joint_sequence_with_object(model_path, joint_sequence, object_name=None, 
                                   object_position=None, simulation_speed=1.0):
    """带物体导入的关节序列运动函数
    
    Args:
        model_path: 模型文件路径
        joint_sequence: 关节序列数组
        object_name: 要导入的物体名称 (None表示不导入物体)
        object_position: 物体位置 [x, y, z]
        simulation_speed: 仿真速度倍数
    """
    # 加载模型
    model = mujoco.MjModel.from_xml_path(model_path)
    model.opt.timestep = 0.002
    
    # 使用第一个关节位置作为初始位置
    initial_positions = joint_sequence[0,:] if joint_sequence.shape[0] > 0 else None
    
    # 创建控制器
    controller = MultiSequenceController(model, kp=100, kd=50, initial_positions=initial_positions)
    
    # 导入物体（如果指定了物体名称和位置）
    if object_name is not None and object_position is not None:
        success = controller.object_importer.place_object(object_name, np.array(object_position))
        if not success:
            print(f"物体导入失败，继续运行仿真但不显示物体")
    
    # 添加所有序列到控制器
    base_duration = 1.0
    for i in range(joint_sequence.shape[0]):
        duration = base_duration / simulation_speed
        controller.add_sequence(joint_sequence[i,:], duration=duration)
    
    print(f"已添加 {len(controller.sequences)} 个动作序列")
    if object_name:
        print(f"已导入物体: {object_name} 位置: {object_position}")
    
    sleep_time = 0.001 / simulation_speed
    
    # 启动仿真
    with mujoco.viewer.launch_passive(controller.model, controller.data) as viewer:
        # 配置相机指向物体位置（如果有物体）
        if object_position is not None:
            viewer.cam.distance = 1.5
            viewer.cam.azimuth = 225
            viewer.cam.elevation = -25
            viewer.cam.lookat = np.array(object_position)
        else:
            # 默认相机配置
            viewer.cam.distance = 2.0
            viewer.cam.azimuth = 120
            viewer.cam.elevation = -20
            viewer.cam.lookat = np.array([0, 0, 0.5])
        
        print("可视化仿真开始!")
        
        try:
            while viewer.is_running():
                if not controller.step():
                    # 所有序列完成
                    # print("动作序列执行完成")
                    # break
                    pass
                    
                viewer.sync()
                time.sleep(max(sleep_time, 0.0001))
                
        except KeyboardInterrupt:
            print("仿真被用户中断")

# 修改加载函数，支持物体导入参数
def load_linker(object_name=None, object_position=None, speed_multiplier=2.0):
    """加载linker手模型并进行可视化仿真
    
    Args:
        object_name: 要导入的物体名称 (None表示不导入物体)
        object_position: 物体位置 [x, y, z]
        speed_multiplier: 仿真速度倍数
    """
    # model_path = 'model/robot/linker_hand/scene.xml'
    model_path = 'model/robot/linker_hand/object_on_table_pos.xml'
    load_dir = 'assets/trajectory/'

    joint_sequence = np.loadtxt(os.path.join(load_dir, "test.txt"))
    new_row = joint_sequence[-1].copy()
    new_row[2] += 0.4
    joint_sequence = np.vstack([joint_sequence, new_row])

    print(f"成功加载轨迹文件，包含 {joint_sequence.shape[0]} 个位置点")
    
    # 调用带物体导入的仿真函数
    move_joint_sequence_with_object(
        model_path, 
        joint_sequence, 
        object_name=object_name,
        object_position=object_position,
        simulation_speed=speed_multiplier
    )

def main():
    """主函数：演示不同物体的导入"""
    
    # 示例1：不导入物体，只运行机械手
    # print("=== 仿真1: 纯机械手运动 ===")
    # load_linker(object_name=None, object_position=None, speed_multiplier=2.0)
    
    # 示例2：导入苹果
    print("\n=== 仿真2: 机械手运动 + 物体 ===")
    load_linker(object_name=None, object_position=[-0.07223, 1.087375, 0.215], speed_multiplier=1.0)
    

if __name__ == "__main__":
    main()