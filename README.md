navigation_explainer ROS package implements a API for explaining local robot navigation, naimely local planners.
Currently teb_local_planner is implemented in the package, but also other local planners can be added.

The teb_local_planner package implements a plugin to the base_local_planner of the 2D navigation stack. The underlying method called Timed Elastic Band locally optimizes the robot's trajectory with respect to trajectory execution time, separation from obstacles and compliance with kinodynamic constraints at runtime.

Refer to http://wiki.ros.org/teb_local_planner for more information and tutorials.

To run the navigation explainer please type: rosrun navigation_explainer_image explainer.py.
