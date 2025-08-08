#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import math
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ModelStates
from mavros_msgs.msg import State

def dist_2d(p1, p2):
    if p1 is None or p2 is None:
        return float('inf')  # Return a large value if either point is None
    return math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)

def is_in_zone(x, y, zone):
    return dist_2d({'x': x, 'y': y}, {'x': zone['x'], 'y': zone['y']}) < zone['r']

class ScoringSystem:
    def __init__(self):
        rospy.init_node('scoring_system')

        # 需要根据实际 Pub_downtown.py 中的坐标进行调整
        self.no_fly_zones = [
            {'x': 1200, 'y': 0, 'r': 200},
            {'x': 1800, 'y': 0, 'r': 200}
        ]
        # 旋翼模式区（以固定翼起始点为圆心，半径100m）
        self.rotor_zone_radius = 100
        self.fixed_start = None
        self.rotor_start = None

        # 状态变量
        self.fixed_landed = False
        self.rotor_landed = False
        self.fixed_landed_time = None
        self.rotor_takeoff_time = None
        self.crashed = False
        self.rotor_crashed = False
        self.always_fixed = True
        self.fixed_mode = []
        self.fixed_arm = []
        self.rotor_arm = []
        self.score1 = 0.0
        self.score2 = 0.0
        self.model_states = None # 场景中所有模型的实时状态信息
        self.scored = {
            'first_man': False,
            'second_man': False,
            'third_man': False,
            'healthy': False,
            'bad': False
        }
        self.scores = {
            'first_man': 0.0,
            'second_man': 0.0,
            'third_man': 0.0,
            'healthy': 0.0,
            'bad': 0.0
        }
        self.task_completed = False  # 标记任务是否完成

        # Subscribers
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback)
        rospy.Subscriber('/standard_vtol_0/mavros/state', State, self.fixed_state_callback)
        rospy.Subscriber('/iris_0/mavros/state', State, self.rotor_state_callback)
        rospy.Subscriber('/zhihang2025/first_man/pose', Pose, self.first_man_callback) # 对应模型 person_red 的实时计算坐标
        rospy.Subscriber('/zhihang2025/second_man/pose', Pose, self.second_man_callback) # 对应模型 person_yellow 的实时计算坐标
        rospy.Subscriber('/zhihang2025/third_man/pose', Pose, self.third_man_callback) # 对应模型 person_white 的实时计算坐标
        rospy.Subscriber('/zhihang2025/iris_healthy_man/pose', Pose, self.healthy_callback) # 对应模型 landing2 的实时计算坐标 + 偏移 (0.5, 0.5)
        rospy.Subscriber('/zhihang2025/iris_bad_man/pose', Pose, self.bad_callback) # 对应模型 landing2 的实时计算坐标 + 偏移 (0.5, 0.5)
        # rospy.Subscriber('/xtdrone/standard_vtol_0/cmd', Pose, self.fixed_cmd_callback)
        # rospy.Subscriber('/xtdrone/iris_0/cmd', Pose, self.rotor_cmd_callback) # 记录命令信息
        # rospy.Subscriber('/zhihang/first_point', Pose, self.first_point_callback) # GPS引导点
        # rospy.Subscriber('/zhihang/downtown', Pose, self.downtown_callback) # 订阅居民区中心

    def model_states_callback(self, msg):
        self.model_states = msg
        names = msg.name
        poses = msg.pose
        # 固定翼A
        if 'standard_vtol_0' in names:
            idx = names.index('standard_vtol_0')
            fx, fy = poses[idx].position.x, poses[idx].position.y
            if self.fixed_start is None:
                self.fixed_start = {'x': fx, 'y': fy}
            for zone in self.no_fly_zones:
                if is_in_zone(fx, fy, zone):
                    self.crashed = True # 0分
                    break
            # 实时检查降落区——如果固定翼未降落在旋翼模式区，则标记为未降落
            if not self.fixed_landed and not self.fixed_arm[-1] if self.fixed_arm else True:
                if self.check_rotor_zone(fx, fy):
                    self.fixed_landed = True # 0分
                    self.fixed_landed_time = rospy.get_time()
        # 四旋翼B
        if 'iris_0' in names:
            idx = names.index('iris_0')
            rx, ry = poses[idx].position.x, poses[idx].position.y
            if self.rotor_start is None:
                self.rotor_start = {'x': rx, 'y': ry}
            for zone in self.no_fly_zones:
                if is_in_zone(rx, ry, zone):
                    self.rotor_crashed = True
                    break
            # 实时检查降落区 —— 这种东西应该不会出错
            # if not self.rotor_landed and not self.rotor_arm[-1] if self.rotor_arm else True:
            #     if self.check_rotor_zone(rx, ry):
            #         self.rotor_landed = True
            #         self.task_completed = True  # 标记任务完成

    def fixed_state_callback(self, msg):
        self.fixed_mode.append(msg.mode)
        self.fixed_arm.append(msg.armed)
        if not msg.armed and not self.fixed_landed:
            self.fixed_landed = True
            self.fixed_landed_time = rospy.get_time()
        if msg.mode not in ['AUTO.LOITER', 'AUTO.MISSION', 'GUIDED', 'AUTO']:
            self.always_fixed = False

    def rotor_state_callback(self, msg):
        self.rotor_arm.append(msg.armed)
        if msg.armed and self.fixed_landed_time and self.rotor_takeoff_time is None:
            self.rotor_takeoff_time = rospy.get_time()
        if not msg.armed:
            self.rotor_landed = True
            if self.rotor_landed and self.check_rotor_zone(self.rotor_start['x'], self.rotor_start['y']) and self.scores['healthy'] and self.scores['bad']:
                self.task_completed = True  # 标记任务完成

    def compare_and_score(self, label, pose, model_name, score_formula, max_dist, score_weight, height_limit=None):
        if self.model_states is None or self.scored[label]:
            return
        names = self.model_states.name
        poses = self.model_states.pose
        if model_name in names:
            idx = names.index(model_name)
            model_pose = poses[idx]
            px, py = pose.position.x, pose.position.y
            mx, my = model_pose.position.x, model_pose.position.y
            D = dist_2d({'x': px, 'y': py}, {'x': mx, 'y': my})
            if height_limit is not None: # 检测第2阶段的高度限制
                height = pose.position.z
                if height > height_limit:
                    print(f"⚠️ {label} 高度 {height:.2f} 超过限制 {height_limit}，不计分")
                    self.scored[label] = True
                    return
            score = score_formula(D, max_dist, score_weight)
            print(f"position ::: px:{px}; py:{py}; mx:{mx}; my:{my}")
            print(f"\n✅ {label} 偏差 {D:.2f} 米，得分 {score:.2f}")
            if label in ['first_man', 'second_man', 'third_man']:
                self.score1 += score
            else:
                self.score2 += score
            self.scored[label] = True
            self.scores[label] = score
            print(f"得分 {self.scores[label]:.2f}")
        else:
            print(f"\n⚠️ 未找到 {label} 对应的模型 {model_name}，无法评分")

    def first_man_callback(self, msg):
        if self.scored['first_man']: # 虽然红色是 1495, -105 但是还是直接读取，统一形式
            return
        self.compare_and_score(
            label='first_man',
            pose=msg,
            model_name='person_red',
            score_formula=lambda D, max_dist, w: (100 - 10 * D) * w if D <= max_dist else 0,
            max_dist=10,
            score_weight=0.1
        )

    def second_man_callback(self, msg):
        if self.scored['second_man']:
            return
        self.compare_and_score(
            label='second_man',
            pose=msg,
            model_name='person_yellow',
            score_formula=lambda D, max_dist, w: (100 - 10 * D) * w if D <= max_dist else 0,
            max_dist=10,
            score_weight=0.1
        )

    def third_man_callback(self, msg):
        if self.scored['third_man']:
            return
        self.compare_and_score(
            label='third_man',
            pose=msg,
            model_name='person_white',
            score_formula=lambda D, max_dist, w: (100 - 10 * D) * w if D <= max_dist else 0,
            max_dist=10,
            score_weight=0.1
        )

    def healthy_callback(self, msg):
        if self.scored['healthy']:
            return
        ps = msg
        ps.position.x += 0.5 # 靶心偏移
        ps.position.y += 0.5
        self.compare_and_score(
            label='healthy',
            pose=ps,
            model_name='landing2',
            score_formula=lambda D, max_dist, w: (100 - 33 * D) * w if D <= max_dist else 0,
            max_dist=3,
            score_weight=0.1,
            height_limit=0.7
        )

    def bad_callback(self, msg):
        if self.scored['bad']:
            return
        ps = msg
        ps.position.x += 0.5 # 靶心偏移
        ps.position.y += 0.5
        self.compare_and_score(
            label='bad',
            pose=ps,
            model_name='landing2',
            score_formula=lambda D, max_dist, w: (100 - 33 * D) * w if D <= max_dist else 0,
            max_dist=3,
            score_weight=0.3,
            height_limit=0.5
        )

    def check_rotor_zone(self, x, y):
        if self.fixed_start is None:
            return False
        return dist_2d({'x': x, 'y': y}, self.fixed_start) <= self.rotor_zone_radius

    def calculate_scores(self):
        print(f"\n✅ 阶段一 SCORE1 = {self.score1:.2f}")
        print(f"✅ 阶段二 SCORE2 = {self.score2:.2f}")
        print(f"🎯 总分 SCORE = {self.score1 + self.score2:.2f}")

    def check_compliance(self):
        # 阶段一禁飞区
        if self.crashed:
            rospy.logerr("❌ 固定翼进入禁飞区，任务终止，SCORE = 0")
            return False
        # 固定翼模式检查
        if not self.always_fixed:
            rospy.logerr("❌ 未保持固定翼模式飞行，SCORE = 0")
            return False
        # 固定翼降落检查
        if not self.fixed_landed:
            rospy.logerr("❌ 固定翼未在旋翼模式区降落，SCORE = 0")
            return False
        # 阶段二禁飞区
        if self.rotor_crashed:
            rospy.logerr("❌ 四旋翼进入禁飞区，任务终止，SCORE2 = 0")
            return False
        # 四旋翼降落检查
        if not self.rotor_landed:
            rospy.logerr("❌ 四旋翼未在旋翼模式区降落，SCORE2 = 0")
            return False
        # 四旋翼起飞时机检查
        if self.rotor_takeoff_time is None or (self.fixed_landed_time and self.rotor_takeoff_time < self.fixed_landed_time):
            rospy.logerr("❌ 四旋翼未在固定翼降落后起飞，SCORE2 = 0")
            return False
        return True

    def run(self):
        rospy.loginfo("Scoring system running...")
        rate = rospy.Rate(1)  # 每秒检查一次任务状态
        while not rospy.is_shutdown():
            if self.task_completed:
                rospy.loginfo("任务完成，输出最终结果...")
                self.calculate_scores()
                if self.check_compliance():
                    rospy.loginfo(f"✅ 阶段一 SCORE1 = {self.score1:.2f}")
                    rospy.loginfo(f"✅ 阶段二 SCORE2 = {self.score2:.2f}")
                    rospy.loginfo(f"🎯 总分 SCORE = {self.score1 + self.score2:.2f}")
                else:
                    rospy.loginfo("❌ 总分 SCORE = 0")
                break
            rate.sleep()  # 等待下一次检查任务状态

if __name__ == '__main__':
    try:
        scoring_system = ScoringSystem()
        scoring_system.run()
    except rospy.ROSInterruptException:
        pass
