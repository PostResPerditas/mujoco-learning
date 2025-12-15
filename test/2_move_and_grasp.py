import mujoco
import mujoco.viewer
import time
import numpy as np
import os

class ImprovedObjectImporter:
    """改进的物体导入器，带接触优化"""
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.available_objects = ['potted_meat_can']
        
    def place_object(self, obj_name, position, quaternion=None):
        """在指定位置放置物体，并确保物理稳定性"""
        if obj_name not in self.available_objects:
            print(f"错误: 物体 '{obj_name}' 不可用")
            return False
            
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        if body_id == -1:
            print(f"错误: 未找到物体 '{obj_name}'")
            return False
            
        # 隐藏所有其他物体
        self.hide_all_objects()
        
        # 设置目标物体位置
        start_idx = self.model.jnt_qposadr[self.model.body_jntadr[body_id]]
        
        # 设置位置和姿态
        self.data.qpos[start_idx:start_idx+3] = position
        
        if quaternion is not None:
            self.data.qpos[start_idx+3:start_idx+7] = quaternion
        else:
            self.data.qpos[start_idx+3:start_idx+7] = np.array([1, 0, 0, 0])
            
        # 重置速度为零
        vel_start_idx = self.model.jnt_dofadr[self.model.body_jntadr[body_id]]
        self.data.qvel[vel_start_idx:vel_start_idx+6] = 0
        
        # 运行几步仿真让物体稳定
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        print(f"已稳定放置物体 '{obj_name}' 到位置 {position}")
        return True
    
    def hide_all_objects(self):
        """隐藏所有物体"""
        hide_pos = np.array([10.0, 10.0, 10.0])
        for obj_name in self.available_objects:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
            if body_id != -1:
                start_idx = self.model.jnt_qposadr[self.model.body_jntadr[body_id]]
                self.data.qpos[start_idx:start_idx+3] = hide_pos

class ImprovedGraspingController:
    """改进的抓取控制器，带柔顺控制"""
    
    def __init__(self, model, kp=15, kd=8, initial_positions=None):  # 降低增益
        self.model = model
        self.data = mujoco.MjData(model)
        
        # 设置初始关节位置
        if initial_positions is not None:
            self.set_initial_positions(initial_positions)
        
        # 使用更柔顺的控制参数
        self.kp = kp
        self.kd = kd
        self.sequences = []
        self.current_sequence_index = 0
        self.current_step = 0
        
        # 初始化物体导入器
        self.object_importer = ImprovedObjectImporter(model, self.data)
        
        # 接触力监控
        self.max_contact_force = 0
        self.contact_force_threshold = 5.0  # 接触力阈值(N)
        
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
        
    def compliant_pd_control(self, target_positions):
        """柔顺PD控制器，带接触力适应"""
        current_contact_force = self.get_max_contact_force()
        
        # 根据接触力自适应调整增益
        adaptive_kp = self.kp
        adaptive_kd = self.kd
        
        if current_contact_force > self.contact_force_threshold:
            # 检测到过大接触力时降低增益
            reduction_factor = max(0.1, 1.0 - (current_contact_force - self.contact_force_threshold) / 10.0)
            adaptive_kp = self.kp * reduction_factor
            adaptive_kd = self.kd * reduction_factor
            if self.current_step % 100 == 0:  # 每100步打印一次警告
                print(f"接触力过大: {current_contact_force:.2f}N, 降低增益至 {reduction_factor:.2f}倍")
        
        for i in range(min(len(target_positions), self.model.nu)):
            error_pos = target_positions[i] - self.data.qpos[i]
            current_vel = self.data.qvel[i] if i < len(self.data.qvel) else 0.0
            torque = adaptive_kp * error_pos - adaptive_kd * current_vel
            
            # 限制扭矩输出
            max_torque = 5.0  # 降低最大扭矩限制
            torque = np.clip(torque, -max_torque, max_torque)
            
            if i < self.model.nu:
                self.data.ctrl[i] = torque
                
    def get_max_contact_force(self):
        """获取最大接触力"""
        max_force = 0.0
        if self.data.ncon > 0:
            for i in range(self.data.ncon):
                if i < len(self.data.efc_force):
                    force_magnitude = abs(self.data.efc_force[i])
                    if force_magnitude > max_force:
                        max_force = force_magnitude
        return max_force
        
    def get_current_target(self):
        if self.current_sequence_index >= len(self.sequences):
            return None
        return self.sequences[self.current_sequence_index]['target_positions']
        
    def step(self):
        """执行单步仿真"""
        target = self.get_current_target()
        if target is None:
            return False
            
        # 使用柔顺控制
        self.compliant_pd_control(target)
        mujoco.mj_step(self.model, self.data)
        
        # 监控接触力
        current_force = self.get_max_contact_force()
        if current_force > self.max_contact_force:
            self.max_contact_force = current_force
        
        self.current_step += 1
        sequence = self.sequences[self.current_sequence_index]
        
        if self.current_step >= sequence['steps']:
            self.current_sequence_index += 1
            self.current_step = 0
            
        return self.current_sequence_index < len(self.sequences)

def improved_move_joint_sequence(model_path, joint_sequence, object_name=None, 
                                object_position=None, simulation_speed=1.0):
    """改进的带物体导入的关节序列运动函数"""
    
    # 加载模型
    model = mujoco.MjModel.from_xml_path(model_path)
    
    # 使用更柔顺的控制参数
    initial_positions = joint_sequence[0,:] if joint_sequence.shape[0] > 0 else None
    controller = ImprovedGraspingController(model, kp=12, kd=6, initial_positions=initial_positions)  # 进一步降低增益
    
    # 导入物体
    # if object_name is not None and object_position is not None:
    #     success = controller.object_importer.place_object(object_name, np.array(object_position))
    #     if not success:
    #         print("物体导入失败，继续运行仿真但不显示物体")
    
    # 添加序列
    base_duration = 1.5  # 增加持续时间使运动更平滑
    for i in range(joint_sequence.shape[0]):
        duration = base_duration / simulation_speed
        controller.add_sequence(joint_sequence[i,:], duration=duration)
    
    print(f"已添加 {len(controller.sequences)} 个动作序列")
    print(f"使用柔顺控制参数: kp={controller.kp}, kd={controller.kd}")
    
    sleep_time = 0.001 / simulation_speed
    
    # 启动仿真
    with mujoco.viewer.launch_passive(controller.model, controller.data) as viewer:
        # 配置相机
        if object_position is not None:
            viewer.cam.distance = 1.5
            viewer.cam.azimuth = 225
            viewer.cam.elevation = -25
            viewer.cam.lookat = np.array(object_position)
        else:
            viewer.cam.distance = 2.0
            viewer.cam.azimuth = 120
            viewer.cam.elevation = -20
            viewer.cam.lookat = np.array([0, 0, 0.5])
        
        print("柔顺控制仿真开始! 监控接触力...")
        
        try:
            while viewer.is_running():
                if not controller.step():
                    # 所有序列完成
                    print(f"仿真完成。最大接触力: {controller.max_contact_force:.2f}N")
                    # 保持仿真运行用于观察
                    keep_alive_start = time.time()
                    while time.time() - keep_alive_start < 3.0:
                        viewer.sync()
                        time.sleep(0.01)
                    break
                    
                # 实时显示接触力信息
                if controller.current_step % 50 == 0:
                    current_force = controller.get_max_contact_force()
                    if current_force > 2.0:  # 只显示较大的接触力
                        print(f"当前接触力: {current_force:.2f}N")
                
                viewer.sync()
                time.sleep(max(sleep_time, 0.0001))
                
        except KeyboardInterrupt:
            print("仿真被用户中断")

# 修改加载函数使用改进的控制器
def improved_load_linker(object_name=None, object_position=None, speed_multiplier=1.5):  # 降低速度倍数
    """使用改进控制器加载linker手模型"""
    model_path = 'model/robot/linker_hand/object_on_table.xml'
    load_dir = 'assets/trajectory/'

    joint_sequence = np.loadtxt(os.path.join(load_dir, "test.txt"))
    print(f"成功加载轨迹文件，包含 {joint_sequence.shape[0]} 个位置点")
    
    # 调用改进的仿真函数
    improved_move_joint_sequence(
        model_path, 
        joint_sequence, 
        object_name=object_name,
        object_position=object_position,
        simulation_speed=speed_multiplier
    )

def main():
    """主函数：演示改进的抓取仿真"""
    print("=== 改进的柔顺抓取仿真 ===")
    improved_load_linker(object_name=None, 
                        object_position=[-0.07223, 1.087375, 0.215], 
                        speed_multiplier=1.5)

if __name__ == "__main__":
    main()