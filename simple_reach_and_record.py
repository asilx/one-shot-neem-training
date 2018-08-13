import math
import json
import rospy
import tf2_geometry_msgs
import tf

from tf import TransformListener
from robosherlock_msgs.srv import RSQueryService
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

import sys
from hsrb_interface import geometry

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS
cv2_img = ''
bridge = CvBridge()
heart_a_message = False

flags.DEFINE_string('map_frame', 'map', '')
flags.DEFINE_string('base_frame', 'base_link', '')
flags.DEFINE_string('sensor_frame', 'map', '')
flags.DEFINE_string('end_effector_frame', 'head_rgbd_sensor_rgb_frame', '')

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
        self.omni_base = robot.get(FLAGS.omni_base)
        self.whole_body = robot.get(FLAGS.whole_body)
        self.gripper = robot.get(FLAGS.gripper)
        self.tf_env = TransformListener()
    
    def detect(self, query='{\"detect\":{\"color\":\"yellow\"}}'):
        try:
            res = self.detector(query)
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
        if len(res.answer) != 1:
            raise ValueError('')
        sol = res.answer[0]
        object_info = json.loads(sol)
        goal = PoseStamped()
        goal.header.frame_id = object_info['header']['frame_id']
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = object_info['pose']['position']['x']
        goal.pose.position.y = object_info['pose']['position']['y']
        goal.pose.position.z = object_info['pose']['position']['z']
        goal.pose.orientation.w = object_info['pose']['orientation']['w']
        goal.pose.orientation.x = object_info['pose']['orientation']['x']
        goal.pose.orientation.y = object_info['pose']['orientation']['y']
        goal.pose.orientation.z = object_info['pose']['orientation']['z']

        self.tf_env.waitForTransform(self.end_effector_frame, self.map_frame, rospy.Time(0),rospy.Duration(4.0))
        goal_reference = self.tf_env.transformPoint("map", goal)
        return goal_reference

    def image_callback(msg):
        print("Received an image!")
        try:
            # Convert your ROS Image message to OpenCV2
            heart_a_message = true
            cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError, e:
            print(e)

    def return_observation(self, image_topic, end_effector_frame, object_pose):
        br = tf.TransformBroadcaster()
        br.sendTransform()
        br.sendTransform((object_pose.pose.position.x, object_pose.pose.position.y, object_pose.pose.position.z), 
            (object_pose.pose.orientation.w, object_pose.pose.orientation.x, object_pose.pose.orientation.y, object_pose.pose.orientation.z), rospy.Time.now(), "obj_of_interest", "map")
        pos, rot= self.tf_env.lookupTransform(end_effector_frame, "obj_of_interest", rospy.Time(0))
        t = tf.Transformer(True, rospy.Duration(10.0))
        pos_speed, rot_euler = t.lookupTwist(end_effector_frame, "obj_of_interest", rospy.Time.now(), rospy.Duration(5))
        quaternion_speed = tf.transformations.quaternion_from_euler(rot_euler[0], rot_euler[1], rot_euler[0])
        sub_image = rospy.Subscriber(image_topic, Image, image_callback)
        while not heart_a_message:
           wait = true
        sub_image.unregister()
        heart_a_message = false
        cv2.resize(cv2_img, (FLAGS.im_width, FLAGS.im_height)) 
        path = FLAG.neems + '/real_data/now.jpg'
        cv2.imwrite(path, cv2_img)
        return cv2_img, pos+rot, pos_speed+quaternion_speed

    def reach(self, target_pose):
         # self.whole_body.move_to_neutral()  # disabled for performance
        self.gripper.set_distance(0.1)
        rospy.loginfo("moving to %s" % target_pose.pose.position)
        gripper_offset = 0.06
        ek_offset = 0.0
        pregrasp_offset = 0.06

        #self.go_cancel()
        #self.param_for_go()
        # self.whole_body.looking_hand_constraint = True  # disabled for performance

        target_cds = {
            "x": target_pose.pose.position.x,
            "y": target_pose.pose.position.y,
            "z": target_pose.pose.position.z + gripper_offset,
            "ei": math.pi,
            "ek": ek_offset,
        }
        moved = False
        print rospy.Time.now()
        try:
            self.whole_body.move_end_effector_pose(
                geometry.pose(**target_cds),
                ref_frame_id=target_pose.header.frame_id)
            print rospy.Time.now()
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
                print rospy.Time.now()
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
                print rospy.Time.now()
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
