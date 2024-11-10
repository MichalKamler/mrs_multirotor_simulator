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

    void sphericalToCartesian(double phi, double theta, double& x, double& y, double& z) const {
        x = sin(theta) * cos(phi);
        y = sin(theta) * sin(phi);
        z = cos(theta);
    }

    // Update points inside a spherical cap
    void updateSphericalCap(double phi_center, double theta_center, double radius) {
        // Convert the center of the spherical cap to Cartesian coordinates
        double x_center, y_center, z_center;
        sphericalToCartesian(phi_center, theta_center, x_center, y_center, z_center);   

        int phi_min = static_cast<int>((std::fmod(phi_center - radius + M_PI, 2 * M_PI) - M_PI) / (2 * M_PI / (phi_size - 1)));
        int phi_max = static_cast<int>((std::fmod(phi_center + radius + M_PI, 2 * M_PI) - M_PI) / (2 * M_PI / (phi_size - 1)));
        int theta_min = static_cast<int>((std::fmod(theta_center - radius + M_PI_2, M_PI)) / (M_PI / (theta_size - 1)));
        int theta_max = static_cast<int>((std::fmod(theta_center + radius + M_PI_2, M_PI)) / (M_PI / (theta_size - 1)));

        // Iterate over all points in the field
        for (int i = phi_min; i < phi_max; i++) {
            for (int j = theta_min; j < theta_max; j++) {
                // Convert grid point (i, j) to spherical coordinates
                double phi = 2 * M_PI * i / field.cols(); // Normalize phi to [0, 2π]
                double theta = M_PI * j / field.rows();  // Normalize theta to [0, π]

                // Convert spherical coordinates (phi, theta) to Cartesian
                double x, y, z;
                sphericalToCartesian(phi, theta, x, y, z);

                // Compute the angle between the point and the center of the spherical cap
                double dot_product = x * x_center + y * y_center + z * z_center;
                double distance = acos(dot_product);  // Angle between the two vectors (in radians)

                // If the distance is less than or equal to the cap radius, update the point
                if (distance <= radius) {
                    field(j, i) = 1;  // Set point to 1
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