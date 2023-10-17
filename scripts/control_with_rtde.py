import rtde_control
import rtde_receive
import time
from recording_upload_server import CaptureServer
import tf.transformations as tft
from math import pi

rtde_c = rtde_control.RTDEControlInterface("192.168.1.13")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.13")

ACTION_DICT = {
	'w': ([0.05, 0., 0., 0., 0., 0., ]),
	's': ([-0.05, 0., 0., 0., 0., 0., ]),
	'a': ([0., 0.05, 0., 0., 0., 0., ]),
	'd': ([0., -0.05, 0., 0., 0., 0., ]),
	# 'q': ([0., 0., 0.05, 0., 0., 0., ]),
	# 'e': ([0., 0., -0.05, 0., 0., 0., ]),
	'i': ([0.,0.,0., 20./180. * pi, 0., 0.,]),
	'k': ([0.,0.,0., -20./180. * pi, 0., 0.,]),
	'j': ([0.,0.,0., 0., 20./180. * pi, 0.,]),
	'l': ([0.,0.,0., 0., -20./180. * pi, 0.,]),
	'u': ([0.,0.,0., 0., 0., 20./180. * pi,]),
	'o': ([0.,0.,0., 0., 0., -20./180. * pi,])

}


# low-level functions

def get_pose():
    return rtde_r.getActualTCPPose()

def move_until_contact(speed = [0, 0, -0.050, 0, 0, 0]):
    rtde_c.moveUntilContact(speed)
    # rtde_c.stopScript()

def move_down(dist = 0.005):
    q = get_pose()
    q[2] -= dist
    rtde_c.moveL(q, 0.01, 0.1)

def move_y(dist=-0.01):
    q = get_pose()
    q[1] += dist
    rtde_c.moveL(q, 0.01, 0.1)

def move_tool_frame(action, tcp_offset = [0., 0., 0.2298, 0., 0., 0.,]):
    pose_tool = get_pose()
    print(pose_tool)
    pose_tool = get_pose()
    rtde_c.setTcp(tcp_offset)
    print(rtde_c.getTCPOffset())
    # pose_tool = rtde_r.getTCPPose()
    # print(pose_tool)
    # offset_base = rtde_c.poseTrans(pose_tool, tcp_offset)
    action_base = rtde_c.poseTrans(pose_tool, action)
    # action_base = rtde_c.poseTrans(pose_tool, action)
    # print(action_base)
    rtde_c.moveL(action_base, 0.1, 0.5)

def get_force():
    return rtde_r.getActualTCPForce()

def zero_force_sensor():
    rtde_c.zeroFtSensor()

def simple_force_fbk_move(move_func, target_force_z):
    # init_force_z = get_force()[2]
    print('init force : {:.3f}N'.format(target_force_z))
    move_func()
    while True:
        current_force_z = get_force()[2]
        print('current force : {:.3f}N'.format(current_force_z))
        if current_force_z < target_force_z - 2.:
            move_down(0.001)
            print('move down!')
        elif current_force_z > target_force_z + 2.:
            move_down(-0.001)
            print("move_up!")
        else:
            break

# high-level functions

def record_parallel_image():
    server = CaptureServer()
    move_down(-0.05)
    move_until_contact()
    for i in range(3):
        move_down(0.001)
        time.sleep(1)
    # time.sleep(1)
    force = get_force()[2]
    for i in range(5):
        simple_force_fbk_move(move_y, force)
        time.sleep(1)
        server.save_data()
    print(get_force())
    move_down(-0.05)
    rtde_c.stopScript()

def test_move_tcp():
    for key in ['i','k', 'j', 'l', 'u', 'o']: # , 'k',
        move_tool_frame(ACTION_DICT.get(key))

test_move_tcp()

# zero_force_sensor()
# print(get_force())
# move_down(-0.05)
# move_y(-0.05)
# print(get_force())

# time.sleep(1)
# for i in range(3):
#     move_down(0.001)
#     time.sleep(1)
# for i in range(3):
#     move_y(0.01)
#     print(get_force())
# print(get_force())


# q = get_pose()
# print(q)