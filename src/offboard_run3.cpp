#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <cmath>
#include <std_msgs/String.h>
#include <std_msgs/Bool.h>



// 目标途径点的GPS坐标
double half_x = 1200.0;
double half_y = -260.0;
double half_z = 21.0;

// 目标GPS引导点的坐标
double target_x = 1500.0;
double target_y = 0.0;
double target_z = 21.0;

// 当前无人机位置
double current_x = 0.0;
double current_y = 0.0;
double current_z = 0.0;

bool position_received = false;
bool gps_info_received = false;
bool arrived_half = false; // 是否到达途径点
bool fixed_wing_mode = false; // 是否已经转换为固定翼模式

mavros_msgs::State current_state;
void state_cb(const mavros_msgs::State::ConstPtr& msg){
    current_state = *msg;
}

void poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    current_x = msg->pose.position.x;
    current_y = msg->pose.position.y;
    current_z = msg->pose.position.z;
    position_received = true;
    //ROS_INFO("Current position: [x: %0.6f, y: %0.6f, z: %0.6f]", current_x, current_y, current_z);
}

void gpsCallback(const geometry_msgs::Pose::ConstPtr& msg)
{
    target_x = msg->position.x;
    target_y = msg->position.y;
    gps_info_received = true;
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    try
    {
        // Convert ROS Image message to OpenCV image
        cv::Mat cv_img = cv_bridge::toCvShare(msg, "bgr8")->image;

        // Display the image
        cv::imshow("camera_image", cv_img);
        cv::waitKey(1);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}
// 计算两点之间的欧几里得距离
double distance(double x1, double y1, double x2, double y2)
{
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "offb_node");
    ros::NodeHandle nh;

    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub = it.subscribe("standard_vtol_0/camera/image_raw", 1, imageCallback);

    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>
            ("standard_vtol_0/mavros/state", 10, state_cb);
    ros::Subscriber local_pos_sub = nh.subscribe<geometry_msgs::PoseStamped>
            ("standard_vtol_0/mavros/local_position/pose", 10, poseCallback);
    ros::Subscriber gps_sub = nh.subscribe("/zhihang/first_point", 10, gpsCallback); 

    ros::Publisher setpoint_pos_pub = nh.advertise<geometry_msgs::PoseStamped>
            ("standard_vtol_0/mavros/setpoint_position/local", 10);
    ros::ServiceClient arming_client = nh.serviceClient<mavros_msgs::CommandBool>
            ("standard_vtol_0/mavros/cmd/arming");
    ros::ServiceClient set_mode_client = nh.serviceClient<mavros_msgs::SetMode>
            ("standard_vtol_0/mavros/set_mode");
    ros::Publisher xtdrone_state = nh.advertise<std_msgs::String>("standard_vtol_0/cmd", 10);

    //the setpoint publishing rate MUST be faster than 2Hz
    ros::Rate rate(20.0);

    // wait for FCU connection
    while(ros::ok() && !current_state.connected){
        ros::spinOnce();
        rate.sleep();
    }

    //send a few setpoints before starting
    geometry_msgs::PoseStamped pose;
    pose.pose.position.z = target_z; // Initial target height

    for(int i = 100; ros::ok() && i > 0; --i){
        setpoint_pos_pub.publish(pose);
        ros::spinOnce();
        rate.sleep();
    }

    mavros_msgs::SetMode offb_set_mode;
    offb_set_mode.request.custom_mode = "OFFBOARD";

    mavros_msgs::CommandBool arm_cmd;
    arm_cmd.request.value = true;

    ros::Time last_request = ros::Time::now();

    // 解锁无人机
    while(ros::ok() && !current_state.armed){
        if( current_state.mode != "OFFBOARD" &&
            (ros::Time::now() - last_request > ros::Duration(5.0))){
            if( set_mode_client.call(offb_set_mode) &&
                offb_set_mode.response.mode_sent){
                ROS_INFO("Offboard enabled");
            }
            last_request = ros::Time::now();
        } else {
            if( !current_state.armed &&
                (ros::Time::now() - last_request > ros::Duration(5.0))){
                if( arming_client.call(arm_cmd) &&
                    arm_cmd.response.success){
                    ROS_INFO("Vehicle armed");
                }
                last_request = ros::Time::now();
            }
        }
        setpoint_pos_pub.publish(pose);
        ros::spinOnce();
        rate.sleep();
    }
    
    while(ros::ok()){
        // 确保起飞到目标高度
        if (current_z < target_z - 0.5)
        {
            pose.pose.position.z = target_z;
        }
        else
        {
            pose.pose.position.z = target_z;
            if (!fixed_wing_mode)
            {
                // 切换到固定翼模式
                std_msgs::String mode_msg;
                mode_msg.data = "plane";
                xtdrone_state.publish(mode_msg);

                ROS_INFO("Switched to fixed-wing mode");
                fixed_wing_mode = true;
            }
        }

        // 导航到目标点
        if (fixed_wing_mode)
        {
            pose.pose.position.x = half_x;
            pose.pose.position.y = half_y;
            pose.pose.position.z = half_z;
        }
        
        double dist_to_half = distance(current_x, current_y, half_x, half_y);
        if (dist_to_half < 3.0) 
        {
            ROS_INFO("Reach target: [x: %0.2f, y: %0.2f, z: %0.2f]", current_x, current_y, current_z);
            arrived_half = true;
        }
        
        // 如果到达途径点
        if (arrived_half) 
        {
            pose.pose.position.x = target_x;
            pose.pose.position.y = target_y;
            pose.pose.position.z = target_z;
        }

        double dist_to_target = distance(current_x, current_y, target_x, target_y);
        if (dist_to_target < 3.0) 
        {
            ROS_INFO("Reach target: [x: %0.2f, y: %0.2f, z: %0.2f]", current_x, current_y, current_z);
            break;
        }

        // 每次循环时更新无人机位置
        setpoint_pos_pub.publish(pose);

        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}

