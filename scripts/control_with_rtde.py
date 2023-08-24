import rtde_control
import rtde_receive
import time
from recording_upload_server import CaptureServer

rtde_c = rtde_control.RTDEControlInterface("192.168.1.13")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.13")


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


# zero_force_sensor()
# print(get_force())
# move_down(-0.05)
# move_y(-0.05)
# print(get_force())
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
# time.sleep(1)
# for i in range(3):
#     move_down(0.001)
#     time.sleep(1)
# for i in range(3):
#     move_y(0.01)
#     print(get_force())
# print(get_force())
rtde_c.stopScript()

# q = get_pose()
# print(q)