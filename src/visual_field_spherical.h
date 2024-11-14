// VisualField.h
#ifndef VISUAL_FIELD_H
#define VISUAL_FIELD_H

#include <Eigen/Dense>

struct V_spherical {

    Eigen::MatrixXi field;
    int phi_size;
    int theta_size;

    // Constructor takes phi and theta as grid resolutions
    V_spherical(int phi, int theta) : field(theta, phi) {
        field.setZero();  // Initialize all points to 0
        phi_size = phi;
        theta_size = theta;
    }

    double phi_to_0_to_2pi(double phi) const {
        if (phi < 0) {
            phi += 2 * M_PI;  // Shift negative angles into [0, 2π)
        }
        return phi;
    }

    double theta_to_0_to_pi(double theta) const { //this theta represents angle from the z axis
        return M_PI_2-theta;
    }

    void sphericalToCartesian(double phi, double theta, double& x, double& y, double& z) const {
        phi = phi_to_0_to_2pi(phi);
        theta = theta_to_0_to_pi(theta);
        x = sin(theta) * cos(phi);
        y = sin(theta) * sin(phi);
        z = cos(theta);
    }

    double phi_int_to_angle(int i){ //returns angle from [-pi, pi], arg is int i that is [0, phi_size-1]
        return (static_cast<double>(i) * 2 * M_PI) / phi_size - M_PI;
    }

    double theta_int_to_angle(int i){ //returns angle from [-pi/2, pi/2], arg is int i that is [0, theta_size-1]
        return (static_cast<double>(i) * M_PI) / theta_size - M_PI_2;
    }

    int phi_angle_to_int(double angle){ //returns int from [0, phi_size-1], arg is double angle from R with 2pi periodicity
        angle = phi_angle_shift(angle) + M_PI; //0 to 2pi
        return static_cast<int>(std::fmod(angle, 2 * M_PI) * (phi_size)/(2 * M_PI)); 
    }

    double phi_angle_shift(double phi){ //shift the phi back to -pi to pi if overflow
        if (phi>=-M_PI && phi<=M_PI){ //phi already ok
            return phi;
        }
        while (phi >= M_PI){
            phi -= 2 * M_PI;
        }
        while (phi <= -M_PI){
            phi += 2 * M_PI;
        }
        return phi;
    }

    int theta_angle_to_int(double angle){ //returns int from [0, theta_size-1], arg is double angle from R
        return std::min(static_cast<int>((sin(angle) + 1) * (theta_size)/(2)), theta_size - 1); 
    }

    // Update points inside a spherical cap
    void updateSphericalCap(double phi_center, double theta_center, double radius) {
        // Convert the center of the spherical cap to Cartesian coordinates
        double x_center, y_center, z_center;
        sphericalToCartesian(phi_center, theta_center, x_center, y_center, z_center);   
        // std::cout << "______________________________________________________________________________________________________________________________________________________________" << std::endl;
        int phi_min = phi_angle_to_int(phi_center-radius);
        int phi_max = phi_angle_to_int(phi_center+radius);
        int theta_min = theta_angle_to_int(std::max(theta_center - radius, -M_PI_2));
        int theta_max = theta_angle_to_int(std::min(theta_center + radius, M_PI_2));
        // std::cout << "phi_min: " << phi_min << ", phi_max: " << phi_max << ", theta_min: " << theta_min << ", theta_max: " << theta_max << std::endl;


        // Adjust phi range if wrapping occurs
        if (phi_min > phi_max) {
            // std::cout << "__________________adjust phimin>phimax________________" << std::endl;
            // Handle wrapping by splitting the loop
            for (int i = phi_min; i < phi_size; i++) {
                for (int j = theta_min; j <= theta_max; j++) {
                    // Convert grid point (i, j) to spherical coordinates
                    double phi = phi_int_to_angle(i);
                    double theta = theta_int_to_angle(j);

                    // Convert spherical coordinates (phi, theta) to Cartesian
                    double x, y, z;
                    sphericalToCartesian(phi, theta, x, y, z);

                    // Compute the angle between the point and the center of the spherical cap
                    double dot_product = x * x_center + y * y_center + z * z_center;
                    double distance = acos(dot_product);  // Angle between the two vectors (in radians)

                    // If the distance is less than or equal to the cap radius, update the point
                    if (distance <= radius) {
                        // std::cout << "__________________in if and setting field from for 11________________" << std::endl;
                        field(j, i) = 1;  // Set point to 1
                    }
                }
            }

            // Loop from 0 to phi_max if phi_min > phi_max (wraparound case)
            for (int i = 0; i <= phi_max; i++) {
                for (int j = theta_min; j <= theta_max; j++) {
                    // Convert grid point (i, j) to spherical coordinates
                    double phi = phi_int_to_angle(i);
                    double theta = theta_int_to_angle(j);

                    // Convert spherical coordinates (phi, theta) to Cartesian
                    double x, y, z;
                    sphericalToCartesian(phi, theta, x, y, z);

                    // Compute the angle between the point and the center of the spherical cap
                    double dot_product = x * x_center + y * y_center + z * z_center;
                    double distance = acos(dot_product);  // Angle between the two vectors (in radians)

                    // If the distance is less than or equal to the cap radius, update the point
                    if (distance <= radius) {                        
                        // std::cout << "__________________in if and setting field from for 22________________" << std::endl;
                        field(j, i) = 1;  // Set point to 1
                    }
                }
            }
        } else {
            // Normal case where phi_min <= phi_max
            // std::cout << "__________________else________________" << std::endl;
            for (int i = phi_min; i <= phi_max; i++) {
                for (int j = theta_min; j <= theta_max; j++) {
                    // Convert grid point (i, j) to spherical coordinates
                    double phi = phi_int_to_angle(i);
                    double theta = theta_int_to_angle(j);

                    // Convert spherical coordinates (phi, theta) to Cartesian
                    double x, y, z;
                    sphericalToCartesian(phi, theta, x, y, z);
                    // std::cout << "phi_center: " << phi_center << ", theta_center: " << theta_center << ", x_center: " << x_center << ", y_center: " << y_center << ", z_center: " << z_center << std::endl;
                    // std::cout << "phi: " << phi << ", theta: " << theta << ", x: " << x << ", y: " << y << ", z: " << z << std::endl;


                    // Compute the angle between the point and the center of the spherical cap
                    double dot_product = x * x_center + y * y_center + z * z_center;
                    double distance = acos(dot_product);  // Angle between the two vectors (in radians)
                    // std::cout << "distance: " << distance << ", dot_product: " << dot_product << std::endl;


                    // If the distance is less than or equal to the cap radius, update the point
                    if (distance <= radius) {
                        // std::cout << "__________________in if and setting field from for 33________________" << std::endl;
                        // std::cout << "_______________________   ahoooooooooooooooooojjjjjjjjj________________" << std::endl;
                        field(j, i) = 1;  // Set point to 1
                    }
                }
            }
        }

    }

    void printField() const {
        std::cout << field << std::endl;
    }

    int integratePhi(int phi_idx) const {
        if (phi_idx >= 0 && phi_idx < field.cols()) {
            return field.col(phi_idx).sum();  // Sums over theta for given phi
        }
        return 0;
    }
};

#endif // VISUAL_FIELD_H