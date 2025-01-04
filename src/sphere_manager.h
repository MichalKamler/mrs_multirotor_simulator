#ifndef SPHERE_MANAGER_H
#define SPHERE_MANAGER_H

#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>
#include <std_msgs/ColorRGBA.h>
#include <vector>
#include <math.h>
#include "visual_field_spherical.h"

class SphereManager {
public:
    SphereManager() : namespace_("default_ns"), frame_id("default_frame") {}
    // Constructor with parameters
    SphereManager(const std::string& ns, const std::string& frame)
        : namespace_(ns), frame_id(frame) {}

    // Function to generate points for a sphere
    void generateSphere(double radius, double x_spawn, double y_spawn, double z_spawn, V_spherical& vi_spherical, double psi, int id) {
        geometry_msgs::Point p;
        std::vector<geometry_msgs::Point> sphere_points;
        std::vector<std_msgs::ColorRGBA> sphere_colors;

        int phi_size = vi_spherical.getPhiSize();  // Assuming this function exists in V_spherical
        int theta_size = vi_spherical.getThetaSize();  // Assuming this function exists in V_spherical

        double cos_psi = cos(psi);
        double sin_psi = sin(psi);

        double delta_theta = M_PI / 2.0 / (theta_size - 1);

        for (int i = 0; i < theta_size; ++i) {
            for (int j = 0; j < phi_size; ++j) {
                if (vi_spherical.field(i, j) == 0) {
                    continue;  // Skip this iteration and go to the next point
                }
                double theta = M_PI - (M_PI * i) / (theta_size - 1);  // Distribute theta values
                double phi = - M_PI + (2 * M_PI * j) / (phi_size);  // Distribute phi values

                double translated_x = radius * sin(theta) * cos(phi);
                double translated_y = radius * sin(theta) * sin(phi);

                double rotated_x = translated_x * cos(-psi) - translated_y * sin(-psi);
                double rotated_y = translated_x * sin(-psi) + translated_y * cos(-psi);

                p.x = rotated_x + x_spawn;
                p.y = rotated_y + y_spawn;

                // p.x = radius * sin(theta) * cos(phi) + x_spawn;
                // p.y = radius * sin(theta) * sin(phi) + y_spawn;
                p.z = radius * cos(theta) + z_spawn;
                sphere_points.push_back(p);

                // Determine color based on the field value at (i, j)
                std_msgs::ColorRGBA color;
                if (vi_spherical.field(i, j) == 1) {
                    // Color for field value 1
                    color.r = 1.0;
                    color.g = 0.0;
                    color.b = 0.0;
                    color.a = 1.0;
                } else {
                    // Color for field value 0
                    color.r = 0.0;
                    color.g = 1.0;
                    color.b = 0.0;
                    color.a = 1.0;
                }
                sphere_colors.push_back(color);
            }
        }

        // Store the sphere's points and colors by id
        spheres_[id] = {sphere_points, sphere_colors};
    }

    void generateCylinder(double radius, double x_spawn, double y_spawn, double z_spawn, Eigen::VectorXd vi_spherical, double psi, int id, int phi_size) {
        geometry_msgs::Point p;
        std::vector<geometry_msgs::Point> cylinder_points;
        std::vector<std_msgs::ColorRGBA> cylinder_colors;

        double cos_psi = cos(psi);
        double sin_psi = sin(psi);

        double delta_phi = 2 * M_PI / phi_size;
        double height = 1.0;

        for (int j = 0; j < phi_size; ++j) {
            if (vi_spherical(j) == 0) {
                continue;  // Skip if this point is not needed based on the field value
            }
            double phi = - M_PI + (2 * M_PI * j) / (phi_size);  // Angle around the circle in the xy-plane

            // Loop through different z levels to create the height of the cylinder
            for (double z = z_spawn; z <= z_spawn + height; z += height / 20.0) {  // Divide height into 20 steps
                double translated_x = radius * cos(phi);
                double translated_y = radius * sin(phi);

                // Rotate the circle in the xy-plane by psi
                double rotated_x = translated_x * cos(-psi) - translated_y * sin(-psi);
                double rotated_y = translated_x * sin(-psi) + translated_y * cos(-psi);

                p.x = rotated_x + x_spawn;
                p.y = rotated_y + y_spawn;
                p.z = z;  // Set z-coordinate for the cylinder's height
                cylinder_points.push_back(p);

                // Color based on the field value
                std_msgs::ColorRGBA color;
                if (vi_spherical(j) == 1) {
                    color.r = 1.0;
                    color.g = 0.0;
                    color.b = 0.0;
                    color.a = 1.0;
                } else {
                    color.r = 0.0;
                    color.g = 1.0;
                    color.b = 0.0;
                    color.a = 1.0;
                }
                cylinder_colors.push_back(color);
            }
        }

        // Store the sphere's points and colors by id
        spheres_[id] = {cylinder_points, cylinder_colors}; //I know but lazy to rename the spheres
    }

    // Function to create a marker for the sphere
    visualization_msgs::Marker createMarker(int id, int action) {
        auto it = spheres_.find(id);
        if (it == spheres_.end()) {
            ROS_WARN("Sphere with id %d not found!", id);
            return visualization_msgs::Marker();
        }

        std::vector<geometry_msgs::Point> points = it->second.first;
        std::vector<std_msgs::ColorRGBA> colors = it->second.second;


        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id;
        marker.header.stamp = ros::Time::now();
        marker.ns = namespace_;
        marker.id = id;
        marker.type = visualization_msgs::Marker::POINTS;
        marker.action = action;

        marker.scale.x = 0.05; // Point width
        marker.scale.y = 0.05; // Point height
        marker.color.a = 1.0;  // Default alpha

        marker.points = points;
        marker.colors = colors;

        return marker;
    }

    void clearSphereData(int id) {
        auto it = spheres_.find(id);
        if (it != spheres_.end()) {
            it->second.first.clear();  // Reset the points vector
            it->second.second.clear();  // Reset the colors vector
        }
    }

    // Function to publish the current marker to delete the sphere
    visualization_msgs::Marker deleteSphere(int id) {
        visualization_msgs::Marker marker = createMarker(id, visualization_msgs::Marker::DELETE);
        clearSphereData(id);
        return marker;
    }

private:
    std::string namespace_;
    std::string frame_id;
    // Map to store spheres' points and colors by id
    std::unordered_map<int, std::pair<std::vector<geometry_msgs::Point>, std::vector<std_msgs::ColorRGBA>>> spheres_;
};

#endif // SPHERE_MANAGER_H