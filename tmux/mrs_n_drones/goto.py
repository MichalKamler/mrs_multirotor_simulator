#!/usr/bin/python3

import rospy
import rosnode
import random
import os

from mrs_msgs.msg import ReferenceStamped as ReferenceStamped

class Goto:

    def __init__(self):

        rospy.init_node('goto', anonymous=True)

        rospy.loginfo('ros not initialized')

        publishers = []
        n_uavs = 5

        rospy.loginfo('setting up publishers')

        for i in range(0, n_uavs):
            publishers.append(rospy.Publisher('/uav{}/control_manager/reference'.format(i+1), ReferenceStamped, queue_size=1))

        rospy.sleep(2.0)

        xs = []
        ys = []
        zs = []
        hdgs = []

        # for i in range(0, n_uavs):

        #     # # random position
        #     xs.append(random.uniform(-20, 20))
        #     ys.append(random.uniform(-20, 20))
        #     zs.append(2)
        #     hdgs.append(random.uniform(-3.14, 3.14))

        #     # particular position
        #     # xs.append(0)
        #     # ys.append(0)
        #     # zs.append(5)
        #     # hdgs.append(0)
        
        # Generate grid positions with center at [0,0]
        grid_size = int(n_uavs ** 0.5)
        offset = 5  # distance between grid points
        half_grid = (grid_size - 1) / 2  # To center the grid around [0, 0]

        for i in range(0, n_uavs):
            row = i // grid_size
            col = i % grid_size
            x_pos = (col - half_grid) * offset
            y_pos = (row - half_grid) * offset

            xs.append(x_pos)
            ys.append(y_pos)
            zs.append(2)  # Fixed altitude for all UAVs
            hdgs.append(0)  # Heading set to 0 for all UAVs, can be randomized or adjusted as needed



        rospy.loginfo('publishing')

        msg = ReferenceStamped();

        for i in range(0, n_uavs):

            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "world_origin"

            msg.reference.position.x = xs[i]
            msg.reference.position.y = ys[i]
            msg.reference.position.z = zs[i]
            msg.reference.heading = hdgs[i]

            publishers[i].publish(msg)


if __name__ == '__main__':
    try:
        goto = Goto()
    except rospy.ROSInterruptException:
        pass
