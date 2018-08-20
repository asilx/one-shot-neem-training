import math
import imageio
import json
import numpy as np
import rospy
import tf2_geometry_msgs
import tf
import tf2_ros

from tf import TransformListener
from robosherlock_msgs.srv import RSQueryService
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
import geometry_msgs.msg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

import sys
import hsrb_interface
from hsrb_interface import geometry
from hsrb_interface import exceptions

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('map_frame', 'map', '')
flags.DEFINE_string('base_frame', 'base_link', '')
flags.DEFINE_string('sensor_frame', 'map', '')
flags.DEFINE_string('end_effector_frame', 'hand_palm_link', '')

flags.DEFINE_string('rs_service', '/RoboSherlock_asil/query', '')
flags.DEFINE_string('image_topic', '/hsrb/head_rgbd_sensor/rgb/image_raw', '')
flags.DEFINE_string('omni_base', 'omni_base', '')
flags.DEFINE_string('whole_body', 'whole_body', '')
flags.DEFINE_string('gripper', 'gripper', '')

class SimpleReachinRecordin(object):
    def __init__(self, config={}):
        self.map_frame = FLAGS.map_frame
        self.base_frame = FLAGS.base_frame
        self.sensor_frame = FLAGS.sensor_frame
        self.end_effector_frame = FLAGS.end_effector_frame

        rospy.wait_for_service(FLAGS.rs_service)
        self.detector = rospy.ServiceProxy(FLAGS.rs_service, RSQueryService)
        self.robot = hsrb_interface.Robot()
        self.omni_base = self.robot.get(FLAGS.omni_base)
        self.whole_body = self.robot.get(FLAGS.whole_body)
        self.gripper = self.robot.get(FLAGS.gripper)
        self.tf_env = TransformListener()
    
    def detect(self, query='{\"detect\":{\"color\":\"yellow\"}}'):
        try:
            res = self.detector(query)
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
        #if len(res.answer) != 1:
        #    raise ValueError('more than one perceived objects that fit with the descriptor')
        sol = res.answer[0]
        object_info = json.loads(sol)
        goal = PoseStamped()
        goal.header.frame_id = object_info['boundingbox']['pose']['header']['frame_id']
        goal.header.stamp = rospy.Time.now() - rospy.Duration(0.1)
        goal.pose.position.x = object_info['boundingbox']['pose']['pose']['position']['x']
        goal.pose.position.y = object_info['boundingbox']['pose']['pose']['position']['y']
        goal.pose.position.z = object_info['boundingbox']['pose']['pose']['position']['z']
        goal.pose.orientation.w = object_info['boundingbox']['pose']['pose']['orientation']['w']
        goal.pose.orientation.x = object_info['boundingbox']['pose']['pose']['orientation']['x']
        goal.pose.orientation.y = object_info['boundingbox']['pose']['pose']['orientation']['y']
        goal.pose.orientation.z = object_info['boundingbox']['pose']['pose']['orientation']['z']

        self.tf_env.waitForTransform(self.end_effector_frame, self.map_frame, rospy.Time(0),rospy.Duration(4.0))
        goal_reference = self.tf_env.transformPose("map", goal)
        return goal_reference

    def return_observation(self, image_topic, end_effector_frame, object_pose):
        br = tf2_ros.StaticTransformBroadcaster()
        static_transformStamped = geometry_msgs.msg.TransformStamped()
  
        static_transformStamped.header.stamp = rospy.Time(0)
        static_transformStamped.header.frame_id = "map"
        static_transformStamped.child_frame_id = "obj_of_interest"
  
        static_transformStamped.transform.translation.x = object_pose.pose.position.x
        static_transformStamped.transform.translation.y = object_pose.pose.position.y
        static_transformStamped.transform.translation.z = object_pose.pose.position.z
  
        static_transformStamped.transform.rotation.x = object_pose.pose.orientation.w
        static_transformStamped.transform.rotation.y = object_pose.pose.orientation.x
        static_transformStamped.transform.rotation.z = object_pose.pose.orientation.y
        static_transformStamped.transform.rotation.w = object_pose.pose.orientation.z
 
        br.sendTransform(static_transformStamped)
        #br.sendTransform((object_pose.pose.position.x, object_pose.pose.position.y, object_pose.pose.position.z),  (object_pose.pose.orientation.w, object_pose.pose.orientation.x, object_pose.pose.orientation.y, object_pose.pose.orientation.z), rospy.Time.now(), "obj_of_interest", "map")
        rospy.sleep(2)
        pos, rot= self.tf_env.lookupTransform(end_effector_frame, "obj_of_interest", rospy.Time.now() - rospy.Duration(0.1))
        #pos_ex, rot_ex= self.tf_env.lookupTransform(end_effector_frame, "obj_of_interest", rospy.Time.now() - rospy.Duration(0.1)) 
        #t = tf.Transformer(True, rospy.Duration(10.0))
        pos_speed, rot_euler = self.tf_env.lookupTwist(end_effector_frame, "obj_of_interest", rospy.Time.now() - rospy.Duration(0.1), rospy.Duration(0.1))
        quaternion_speed = tf.transformations.quaternion_from_euler(rot_euler[0], rot_euler[1], rot_euler[0])
        image_msg = rospy.wait_for_message(image_topic, Image, 3)
        bridge = CvBridge()
        try:
            # Convert your ROS Image message to OpenCV2
            cv2_img = bridge.imgmsg_to_cv2(image_msg , "bgr8")
        except CvBridgeError, e:
            print(e)
        else:
            cv2_img = cv2.resize(cv2_img, (FLAGS.im_width, FLAGS.im_height)) 
            path = FLAGS.neems + 'real_data/now.jpg'
            cv2.imwrite(path, cv2_img)
       
        O = np.array(cv2_img)[:, :, :3]
        O = np.transpose(O, [2, 1, 0]) # transpose to mujoco setting for images
        O = O.reshape(1, -1) / 255.0 # normalize

        pos.extend(rot)
        pos_speed = list(pos_speed)
        pos_speed.extend(list(quaternion_speed)) 
        return cv2_img, pos,pos_speed 

    def reach(self, target_pose):
         # self.whole_body.move_to_neutral()  # disabled for performance
        self.gripper.set_distance(0.1)
        rospy.loginfo("moving to %s" % target_pose)
        gripper_offset = 0.06
        ek_offset = 0.0
        pregrasp_offset = 0.06

        #self.go_cancel()
        #self.param_for_go()
        # self.whole_body.looking_hand_constraint = True  # disabled for performance

        target_cds = {
            "x": target_pose[0][0][0],
            "y": target_pose[0][0][1],
            "z": target_pose[0][0][2] + gripper_offset,
            "ei": math.pi,
            "ek": ek_offset,
        }
        moved = False
        print rospy.Time.now()  - rospy.Duration(0.1)
        try:
            self.whole_body.move_end_effector_pose(
                geometry.pose(**target_cds),
                ref_frame_id="obj_of_interest")
            print rospy.Time.now()  - rospy.Duration(0.1)
            moved = True
        except exceptions.MotionPlanningError as e:
            rospy.logerr(e)
            rospy.logerr(traceback.format_exc())
            rospy.logwarn("Failed to plan pick motion from {0} to {1} from {2}".format(
                self.whole_body.joint_positions,
                target_cds,
                target_pose.header.frame_id))
        if not moved:
            self.whole_body.move_to_neutral(wait=False, move_head=False)
            rospy.logwarn("fallback to circle")
            try:
                # take some distance first, and then reach the object
                # FIXME: how to prove that it is possible
                #        to solve the 2 movements before sending commands?
                target_cds["z"] -= gripper_offset
                self.whole_body.move_end_effector_on_circle(
                    center=geometry.pose(**target_cds),
                    distance=pregrasp_offset + gripper_offset,
                    ref_frame_id=target_pose.header.frame_id)
                self.whole_body.move_end_effector_pose(
                    pose=geometry.pose(z=pregrasp_offset),
                    ref_frame_id=self.end_effector_frame)
                print rospy.Time.now()  - rospy.Duration(0.1)
                moved = True
            except exceptions.MotionPlanningError as e:
                    rospy.logerr(e)
                    rospy.logerr(traceback.format_exc())

        if not moved:
            rospy.logwarn("fallback to rot")
            target_cds["z"] += gripper_offset
            target_cds["ek"] += math.pi/2.
            try:
                self.whole_body.move_end_effector_pose(
                    geometry.pose(**target_cds),
                    ref_frame_id=target_pose.header.frame_id)
                print rospy.Time.now()  - rospy.Duration(0.1)
                moved = True
            except exceptions.MotionPlanningError as e:
                rospy.logerr(e)
                rospy.logerr(traceback.format_exc())

        if not moved:
            return False, {}

        # self.whole_body.looking_hand_constraint = False  # disabled for performance
        #self.param_for_normal()
        #self.gripper.apply_force(0.6, True)
        rospy.sleep(3)  # omajinai
        # self.whole_body.move_to_neutral()  # disabled for performance
        return True, {"target_spot": "box"}
def main():
    robot = SimpleReachinRecordin()
    obj = robot.detect()
    robot.reach(obj)

if __name__ == "__main__":
    rospy.init_node('simple_reach_and_record')
    rospy.sleep(5.0)
    main()
    rospy.spin()
